# main.py

from player import Player
from liar_game import LiarGame

def main():
    print("라이어 게임에 오신 것을 환영합니다!")
    try:
        total_players = int(input("총 플레이어 수를 입력하세요 (최소 3명): "))
    except ValueError:
        print("숫자를 입력해주세요.")
        return

    if total_players < 3:
        print("플레이어 수가 최소 인원(3명)에 미치지 않습니다. 게임을 종료합니다.")
        return

    human_name = input("당신의 이름을 입력하세요: ")
    players = [Player(human_name, is_human=True)]

    # 나머지 플레이어는 AI로 생성합니다.
    for i in range(1, total_players):
        ai_name = f"AI_{i+1}"
        players.append(Player(ai_name))

    game = LiarGame(players)
    game.play_game()

if __name__ == "__main__":
    main()
