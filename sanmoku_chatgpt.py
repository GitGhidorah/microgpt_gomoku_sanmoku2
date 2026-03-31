import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

print("=== START ===")
print("FILE:", __file__)
print("CWD :", os.getcwd())

# ======================
# 勝敗判定
# ======================
def check_win(board, player):
    wins = [
        [0,1,2],[3,4,5],[6,7,8],
        [0,3,6],[1,4,7],[2,5,8],
        [0,4,8],[2,4,6]
    ]
    return any(all(board[i]==player for i in w) for w in wins)

def is_draw(board):
    return np.all(board != 0)

# ======================
# Minimax（キャッシュ付き）
# ======================
memo = {}

def minimax(board, player):
    key = (tuple(board), player)
    if key in memo:
        return memo[key]

    if check_win(board, 2):
        return 1
    if check_win(board, 1):
        return -1
    if is_draw(board):
        return 0

    moves = np.where(board == 0)[0]

    if player == 2:
        best = -999
        for m in moves:
            board[m] = 2
            best = max(best, minimax(board,1))
            board[m] = 0
    else:
        best = 999
        for m in moves:
            board[m] = 1
            best = min(best, minimax(board,2))
            board[m] = 0

    memo[key] = best
    return best

# ======================
# 最善手（★一意に固定）
# ======================
def best_move(board):
    moves = np.where(board == 0)[0]
    best_score = -999
    best_moves = []

    for m in moves:
        board[m] = 2
        score = minimax(board,1)
        board[m] = 0

        if score > best_score:
            best_score = score
            best_moves = [m]
        elif score == best_score:
            best_moves.append(m)

    # ★これが収束の鍵（常に同じ手を選ぶ）
    return min(best_moves) if best_moves else None

# ======================
# MicroGPT（改良版）
# ======================
class MicroGPT(nn.Module):
    def __init__(self, vocab_size=3, emb_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)

        self.q = nn.Linear(emb_dim, emb_dim)
        self.k = nn.Linear(emb_dim, emb_dim)
        self.v = nn.Linear(emb_dim, emb_dim)

        # ★重要：flattenして位置情報を保持
        self.fc = nn.Linear(9 * emb_dim, 9)

    def forward(self, x):
        e = self.embed(x)

        Q = self.q(e)
        K = self.k(e)
        V = self.v(e)

        attn = torch.matmul(Q, K.transpose(-2,-1)) / np.sqrt(Q.size(-1))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)

        # ★ここが重要（meanしない）
        out = out.reshape(out.size(0), -1)

        return self.fc(out)

# ======================
# データ生成
# ======================
def generate_data(n=800):
    print("=== データ生成 ===")
    X, Y = [], []

    for i in range(n):
        if i % 100 == 0:
            print(f"data gen: {i}/{n}")

        board = np.zeros(9, dtype=int)

        for _ in range(5):
            empty = np.where(board == 0)[0]
            if len(empty) == 0:
                break

            # ユーザ（ゆるい）
            move = random.choice(empty)
            board[move] = 1
            if check_win(board,1):
                break

            empty = np.where(board == 0)[0]
            if len(empty) == 0:
                break

            # AI最善手
            m = best_move(board)
            if m is None:
                break

            X.append(board.copy())
            Y.append(m)

            board[m] = 2
            if check_win(board,2):
                break

    print("データ数:", len(X))
    return torch.tensor(np.array(X)), torch.tensor(np.array(Y))

# ======================
# 学習
# ======================
def train():
    X, Y = generate_data()

    model = MicroGPT()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    print("=== 学習開始 ===")
    start = time.time()

    for epoch in range(2000):
        logits = model(X)
        loss = F.cross_entropy(logits, Y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # ★毎回進捗表示
        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            acc = (pred == Y).float().mean().item()

        elapsed = time.time() - start
        print(f"[{epoch:03d}] loss={loss.item():.4f} acc={acc:.3f} time={elapsed:.1f}s")

    print("=== 学習完了 ===")
    return model

# ======================
# pygame
# ======================
def run_game(model):
    pygame.init()
    size = 300
    screen = pygame.display.set_mode((size,size))
    pygame.display.set_caption("TicTacToe GPT")

    cell = size // 3
    board = np.zeros(9, dtype=int)

    def draw():
        screen.fill((255,255,255))

        for i in range(1,3):
            pygame.draw.line(screen,(0,0,0),(0,i*cell),(size,i*cell),2)
            pygame.draw.line(screen,(0,0,0),(i*cell,0),(i*cell,size),2)

        for i in range(9):
            x = (i%3)*cell + cell//2
            y = (i//3)*cell + cell//2

            if board[i] == 1:
                pygame.draw.circle(screen,(255,0,0),(x,y),30)
            elif board[i] == 2:
                pygame.draw.circle(screen,(0,0,255),(x,y),30)

        pygame.display.flip()

    def ai_move():
        x = torch.tensor(board).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=-1).numpy()[0]

        # 空きマスだけ
        for i in range(9):
            if board[i] != 0:
                probs[i] = 0

        if probs.sum() == 0:
            return

        move = np.argmax(probs)
        board[move] = 2

    running = True
    draw()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                x,y = pygame.mouse.get_pos()
                idx = (y//cell)*3 + (x//cell)

                if board[idx] == 0:
                    board[idx] = 1

                    if not check_win(board,1):
                        ai_move()

                draw()

    pygame.quit()

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    print("=== MAIN ===")
    model = train()
    print("=== GAME ===")
    run_game(model)