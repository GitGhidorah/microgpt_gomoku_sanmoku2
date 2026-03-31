# tic_tac_toe_microgpt_fast.py
# pip install pygame torch

import pygame
import sys
import random
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from functools import lru_cache

# -------------------------
# ルール・盤面まわり
# -------------------------

EMPTY = 0
USER = 1   # 先手（○）
AI = -1    # 後手（×）

def check_winner(board):
    lines = [
        (0,1,2),(3,4,5),(6,7,8),
        (0,3,6),(1,4,7),(2,5,8),
        (0,4,8),(2,4,6)
    ]
    for a,b,c in lines:
        s = board[a] + board[b] + board[c]
        if s == 3*USER:
            return USER
        if s == 3*AI:
            return AI
    if all(x != EMPTY for x in board):
        return 0  # 引き分け
    return None   # 続行

def legal_moves(board):
    return [i for i,x in enumerate(board) if x == EMPTY]

def is_terminal(board):
    return check_winner(board) is not None

# -------------------------
# 高速 minimax（キャッシュ付き）
# -------------------------

@lru_cache(maxsize=None)
def minimax_cached(board_tuple, player):
    board = list(board_tuple)
    winner = check_winner(board)
    if winner == AI:
        return 1, None
    elif winner == USER:
        return -1, None
    elif winner == 0:
        return 0, None

    best_move = None
    if player == AI:
        best_score = -1e9
        for m in legal_moves(board):
            board[m] = AI
            score, _ = minimax_cached(tuple(board), USER)
            board[m] = EMPTY
            if score > best_score:
                best_score = score
                best_move = m
        return best_score, best_move
    else:
        best_score = 1e9
        for m in legal_moves(board):
            board[m] = USER
            score, _ = minimax_cached(tuple(board), AI)
            board[m] = EMPTY
            if score < best_score:
                best_score = score
                best_move = m
        return best_score, best_move

def best_move_for_ai(board):
    return minimax_cached(tuple(board), AI)[1]

# -------------------------
# microGPT っぽい小さなモデル
# -------------------------

class TinyMicroGPT(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.d_model = d_model

        self.input_dim = 28
        self.embed = nn.Linear(self.input_dim, d_model)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_attn = nn.Linear(d_model, d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )

    def forward(self, x):
        h = self.embed(x)

        q = self.W_q(h)
        k = self.W_k(h)
        v = self.W_v(h)

        scores = (q * k).sum(dim=-1, keepdim=True) / math.sqrt(self.d_model)
        attn = torch.softmax(scores, dim=-1)
        context = attn * v

        h2 = self.out_attn(context) + h
        logits = self.mlp(h2)
        return logits

def board_to_input(board, player_turn):
    vec = []
    for v in board:
        if v == -1:
            vec.extend([1,0,0])
        elif v == 0:
            vec.extend([0,1,0])
        else:
            vec.extend([0,0,1])
    vec.append(1 if player_turn == USER else -1)
    return torch.tensor(vec, dtype=torch.float32)

# -------------------------
# 学習データ生成（高速）
# -------------------------

def generate_dataset(n_samples=500):
    states = []
    moves = []
    for _ in range(n_samples):
        board = [EMPTY]*9
        num_moves = random.randint(0, 4)  # 浅めにして高速化
        player = USER
        for _ in range(num_moves):
            if is_terminal(board):
                break
            ms = legal_moves(board)
            if not ms:
                break
            m = random.choice(ms)
            board[m] = player
            player = AI if player == USER else USER

        if is_terminal(board):
            continue

        move = best_move_for_ai(board)
        if move is None:
            continue

        states.append(board[:])
        moves.append(move)

    return states, moves

# -------------------------
# 学習ループ
# -------------------------

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMicroGPT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("データ生成中...")
    states, moves = generate_dataset(800)
    print(f"生成した局面数: {len(states)}")

    X = torch.stack([board_to_input(s, AI) for s in states])
    y = torch.tensor(moves, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    epochs = 150
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for i, (bx, by) in enumerate(loader):
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i+1) % 5 == 0:
                print(f"[epoch {epoch}/{epochs}] step {i+1}/{len(loader)} loss={loss.item():.4f}")

        print(f"==> epoch {epoch} avg_loss={total_loss/len(loader):.4f}")

    return model.to("cpu")

# -------------------------
# Pygame での対局
# -------------------------

WIDTH, HEIGHT = 300, 300
LINE_WIDTH = 4
CELL_SIZE = WIDTH // 3

WHITE = (255,255,255)
BLACK = (0,0,0)
BLUE  = (50,50,255)
RED   = (255,50,50)

def draw_board(screen, board):
    screen.fill(WHITE)
    pygame.draw.line(screen, BLACK, (CELL_SIZE,0), (CELL_SIZE,HEIGHT), LINE_WIDTH)
    pygame.draw.line(screen, BLACK, (CELL_SIZE*2,0), (CELL_SIZE*2,HEIGHT), LINE_WIDTH)
    pygame.draw.line(screen, BLACK, (0,CELL_SIZE), (WIDTH,CELL_SIZE), LINE_WIDTH)
    pygame.draw.line(screen, BLACK, (0,CELL_SIZE*2), (WIDTH,CELL_SIZE*2), LINE_WIDTH)

    for i, v in enumerate(board):
        x = (i % 3) * CELL_SIZE
        y = (i // 3) * CELL_SIZE
        cx = x + CELL_SIZE//2
        cy = y + CELL_SIZE//2
        if v == USER:
            pygame.draw.circle(screen, BLUE, (cx,cy), CELL_SIZE//3, LINE_WIDTH)
        elif v == AI:
            pygame.draw.line(screen, RED, (x+20,y+20), (x+CELL_SIZE-20,y+CELL_SIZE-20), LINE_WIDTH)
            pygame.draw.line(screen, RED, (x+CELL_SIZE-20,y+20), (x+20,y+CELL_SIZE-20), LINE_WIDTH)

def ai_move_with_model(board, model):
    model.eval()
    with torch.no_grad():
        x = board_to_input(board, AI).unsqueeze(0)
        logits = model(x).squeeze(0)
        mask = torch.full_like(logits, float("-inf"))
        for m in legal_moves(board):
            mask[m] = 0.0
        move = int(torch.argmax(logits + mask).item())
    return move

def main():
    print("microGPT風モデルを学習します...")
    model = train_model()
    print("学習完了。ゲームを開始します。")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("3目並べ - ユーザ先手 / AI後手(microGPT)")

    clock = pygame.time.Clock()

    board = [EMPTY]*9
    running = True
    user_turn = True
    game_over = False
    result_text = ""

    font = pygame.font.SysFont(None, 36)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if not game_over and user_turn and event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                col = x // CELL_SIZE
                row = y // CELL_SIZE
                idx = row*3 + col
                if board[idx] == EMPTY:
                    board[idx] = USER
                    user_turn = False

        if not game_over and not user_turn:
            time.sleep(0.3)
            move = ai_move_with_model(board, model)
            if move is not None and board[move] == EMPTY:
                board[move] = AI
            user_turn = True

        winner = check_winner(board)
        if winner is not None and not game_over:
            game_over = True
            if winner == USER:
                result_text = "あなたの勝ち！"
            elif winner == AI:
                result_text = "AIの勝ち！"
            else:
                result_text = "引き分け"

        draw_board(screen, board)

        if game_over:
            text_surf = font.render(result_text + " (閉じるには×)", True, BLACK)
            rect = text_surf.get_rect(center=(WIDTH//2, HEIGHT//2))
            screen.blit(text_surf, rect)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
