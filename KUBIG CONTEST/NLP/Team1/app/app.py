import streamlit as st
from player import Player
from liar_game import LiarGame
import random
from ai_utils_bert import compute_secret_embeddings
import time

# Streamlit 페이지 설정
st.set_page_config(page_title="라이어 게임", page_icon="🎭")

# 스타일 추가
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    
    /* 기본 배경 및 텍스트 색상 */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* 기본 텍스트 색상 설정 */
    .stMarkdown, .stText, h1, h2, h3, h4, h5, p, span {
        color: white !important;
    }
    
    /* 입력 필드 스타일링 수정 */
    div[data-baseweb="input"] {
        background-color: rgba(0, 0, 0, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 10px !important;
    }
    
    div[data-baseweb="input"] > input {
        color: #ffffff !important;
        background-color: transparent !important;
        padding: 10px !important;
        font-size: 16px !important;
    }

    /* 입력 필드 placeholder 스타일 */
    div[data-baseweb="input"] > input::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }

    /* 입력 필드 포커스 효과 */
    div[data-baseweb="input"]:focus-within {
        background-color: rgba(0, 0, 0, 0.7) !important;
        border-color: rgba(255, 75, 75, 0.5) !important;
        box-shadow: 0 0 10px rgba(255, 75, 75, 0.3) !important;
    }

    /* 입력 필드 hover 효과 */
    div[data-baseweb="input"]:hover {
        background-color: rgba(0, 0, 0, 0.6) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
    }

    /* 텍스트 입력 시 스타일 */
    div[data-baseweb="input"] > input:not(:placeholder-shown) {
        background-color: rgba(0, 0, 0, 0.7) !important;
        color: #ffffff !important;
    }
            
    /* 설명 입력 카드 스타일 */
    .explanation-input-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* 네온 효과 애니메이션 */
    @keyframes neonGlow {
        0% { box-shadow: 0 0 5px rgba(255, 75, 75, 0.2), 0 0 8px rgba(255, 75, 75, 0.2); }
        50% { box-shadow: 0 0 8px rgba(255, 75, 75, 0.3), 0 0 12px rgba(255, 75, 75, 0.3); }
        100% { box-shadow: 0 0 5px rgba(255, 75, 75, 0.2), 0 0 8px rgba(255, 75, 75, 0.2); }
    }
    
    @keyframes textGlow {
        0% { text-shadow: 0 0 3px rgba(255, 255, 255, 0.3); }
        50% { text-shadow: 0 0 5px rgba(255, 255, 255, 0.5); }
        100% { text-shadow: 0 0 3px rgba(255, 255, 255, 0.3); }
    }
    
    /* 기본 카드 스타일 */
    .game-card {
        background: rgba(255, 255, 255, 0.05);  /* 배경색을 더 투명하게 */
        border-radius: 25px;
        padding: 30px;
        margin: 20px 0;
        color: white;
        position: relative;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
    }
    
    /* 역할 카드 스타일 */
    .role-card {
        font-family: 'Orbitron', sans-serif;
        color: white;
        text-align: center;
        padding: 30px 20px;
        border-radius: 25px;
        margin: 20px auto;
        max-width: 500px;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
        letter-spacing: 0.5px;
    }
    
    .role-card.liar {
        background: linear-gradient(135deg, #ff4b4b 0%, #800000 100%);
        animation: neonGlow 2s infinite;
    }
    
    .role-card.citizen {
        background: linear-gradient(135deg, #4b4bff 0%, #000080 100%);
        animation: neonGlow 2s infinite;
    }
    
    /* 역할 텍스트 스타일 */
    .role-text {
        font-size: min(2.5em, 8vw);  /* 반응형 폰트 크기 */
        font-weight: 700;
        margin: 15px 0;
        padding: 0 10px;  /* 좌우 여백 추가 */
        line-height: 1.2;  /* 줄 간격 조정 */
        white-space: nowrap;  /* 줄바꿈 방지 */
        animation: textGlow 3s infinite;
    }
    
    /* 부제목 스타일 수정 */
    .sub-text {
        font-size: min(1.3em, 5vw);  /* 반응형 폰트 크기 */
        margin: 12px 0;
        opacity: 0.95;
        line-height: 1.3;
        padding: 0 15px;  /* 좌우 여백 추가 */
    }
    
    /* 라운드 표시 스타일 수정 */
    .round-indicator {
        font-size: min(1.2em, 4vw);  /* 반응형 폰트 크기 */
        padding: 8px 15px;
        border-radius: 12px;
        top: 15px;
        left: 15px;
    }
    
    /* 모바일 최적화 미디어 쿼리 */
    @media (max-width: 768px) {
        .role-card {
            padding: 25px 15px;
            margin: 15px auto;
        }
        
        .progress-bar {
            margin: 15px 0 10px 0;
        }
        
        .icon-circle {
            width: 30px;
            height: 30px;
            font-size: 1.1em;
            margin: 10px auto;
        }
    }
    
    /* 라운드 표시 */
    .round-indicator {
        position: absolute;
        top: 20px;
        left: 20px;
        background: rgba(255, 255, 255, 0.1);
        padding: 10px 20px;
        border-radius: 15px;
        font-size: 1.2em;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(135deg, #ff4b4b 0%, #800000 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 15px;
        font-weight: bold;
        font-size: 1.2em;
        transition: all 0.3s ease;
        animation: neonGlow 2s infinite;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(255, 75, 75, 0.5);
    }
    
    /* 프로그레스 바 */
    .progress-bar {
        width: 100%;
        height: 4px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 2px;
        margin: 20px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff4b4b, #800000);
        border-radius: 2px;
        transition: width 0.3s ease;
    }
    
    /* 설명 단계 스타일 */
    .explanation-card {
        background: #121212;
        border-radius: 25px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }
    
    /* 플레이어 턴 표시 */
    .player-turn {
        font-family: 'Orbitron', sans-serif;
        color: white;
        background: linear-gradient(135deg, #4b4bff 0%, #000080 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        animation: neonGlow 2s infinite;
    }
    
    /* 설명 박스 */
    .description-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
    }
    
    /* 힌트 박스 */
    .hint-box {
        background: linear-gradient(135deg, #ffd700 0%, #ff8c00 100%);
        color: black;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        font-weight: bold;
        animation: neonGlow 2s infinite;
    }
    
    /* 투표 단계 스타일 */
    .vote-card {
        background: #121212;
        border-radius: 25px;
        padding: 30px;
        margin: 20px 0;
        color: white;
    }
    
    /* 아이콘 스타일 */
    .icon-circle {
        width: 40px;
        height: 40px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 10px auto;
        font-size: 1.5em;
    }
    
    /* 승자 발표 스타일 */
    .winner-card {
        background: linear-gradient(135deg, #FFD700 0%, #FF8C00 100%);
        color: black;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin: 20px 0;
        animation: neonGlow 2s infinite;
    }
    
    /* 스코어보드 스타일 */
    .score-box {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* 선택 입력 필드 스타일 */
    .stSelectbox {
        color: white;
        background-color: rgba(255, 255, 255, 0.1);
    }
    
    /* 사이드바 스타일 */
    .sidebar .sidebar-content {
        background: #121212 !important;
        opacity: 1 !important;;
    }
    </style>
""", unsafe_allow_html=True)
    
# 게임 정보 표시 함수
def display_game_info():
    game = st.session_state.game
    if game and hasattr(game, 'chosen_topic'):
        with st.sidebar:
            st.markdown("""
                <div class="game-card">
                    <h3>게임 정보</h3>
                    <p>주제: {topic}</p>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {progress}%;"></div>
                    </div>
                </div>
            """.format(
                topic=game.chosen_topic,
                progress=(game.current_round/game.total_rounds)*100
            ), unsafe_allow_html=True)
            
            st.markdown("""
                <div class="game-card">
                    <h3>플레이어 점수</h3>
                    {scores}
                </div>
            """.format(
                scores="".join([f"<p>{p.name}: {p.score}점</p>" for p in game.players])
            ), unsafe_allow_html=True)

def display_role_card(game, is_liar, secret_word=None):
    role_class = "liar" if is_liar else "citizen"
    round_text = f"{game.current_round}/{game.total_rounds}"
    
    card_html = f"""
        <div class="role-card {role_class}">
            <div class="round-indicator">라운드 {round_text}</div>
            <div class="role-text">
                {'당신은 라이어입니다!' if is_liar else '당신은 시민입니다!'}
            </div>
            <div class="sub-text">
                {'' if is_liar else f'제시어: {secret_word}'}
            </div>
            <div class="sub-text">
                {'라이어를 들키지 마세요!' if is_liar else '라이어를 찾으세요!'}
            </div>
            <div class="icon-circle">?</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {(game.current_round/game.total_rounds)*100}%;"></div>
            </div>
        </div>
    """
    return card_html

# 세션 상태 초기화
if 'initialized' not in st.session_state:
    st.session_state.game = None
    st.session_state.game_phase = 'setup'
    st.session_state.descriptions = {}
    st.session_state.current_player_idx = 0
    st.session_state.secret_word = None
    st.session_state.chosen_topic = None
    st.session_state.players_order = None
    st.session_state.votes = {}
    st.session_state.round_data_initialized = False
    st.session_state.liar_word_prediction = None
    st.session_state.initialized = True
    st.session_state.liar_guess_made = False

st.title("라이어 게임\n ##### 🎭난 진짜 라이어 아님. | Team 장어구이")

# 게임 초기 설정
if st.session_state.game_phase == 'setup':
    total_players = st.number_input("총 플레이어 수를 입력하세요 (최소 3명)", min_value=3, value=3)
    human_name = st.text_input("당신의 이름을 입력하세요")
    st.info("tip. 중간 점수를 확인하고 싶다면 사이드바를 확인하세요!")
    
    if st.button("게임 시작하기") and human_name:
        with st.spinner("🎲 게임을 준비하고 있습니다..."):
            players = [Player(human_name, is_human=True)]
            for i in range(1, total_players):
                players.append(Player(f"AI_{i}"))
            st.session_state.game = LiarGame(players)
            st.session_state.game_phase = 'role_reveal'

            time.sleep(5)
            st.info("😉 거의 다 되었습니다...")
            time.sleep(1)

        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# 역할 공개 단계
elif st.session_state.game_phase == 'role_reveal':
    game = st.session_state.game
    
    if not st.session_state.round_data_initialized:
        # 기존 초기화 로직
        game.assign_roles()
        chosen_topic = random.choice(list(game.topics.keys()))
        secret_word = random.choice(game.topics[chosen_topic])
        game.chosen_topic = chosen_topic
        st.session_state.secret_word = secret_word
        
        # 주제별 임베딩 초기화
        game.object_word_embeddings = compute_secret_embeddings(list(game.topics["object"]))
        game.food_word_embeddings = compute_secret_embeddings(list(game.topics["food"]))
        game.job_word_embeddings = compute_secret_embeddings(list(game.topics["job"]))
        game.place_word_embeddings = compute_secret_embeddings(list(game.topics["place"]))
        game.character_word_embeddings = compute_secret_embeddings(list(game.topics["character"]))
        
        players_order = game.players.copy()
        random.shuffle(players_order)
        if players_order[0].is_liar:
            liar_player = players_order.pop(players_order.index(game.liar))
            insert_position = random.randint(1, len(players_order))
            players_order.insert(insert_position, liar_player)
        st.session_state.players_order = players_order
        st.session_state.round_data_initialized = True
    
    # 역할 카드 표시
    human_player = next(p for p in game.players if p.is_human)
    is_liar = human_player.is_liar
    role_class = "liar" if is_liar else "citizen"
    progress_width = (game.current_round/game.total_rounds) * 100
    
    # 라이어와 시민에 따라 다른 내용 준비
    if is_liar:
        role_content = f"""
            <div class="role-card liar">
                <div class="round-indicator">라운드 {game.current_round}/{game.total_rounds}</div>
                <div class="role-text">
                    <br>
                    당신은 라이어입니다!
                </div>
                <div class="sub-text">주제: {game.chosen_topic}</div>
                <div class="sub-text">라이어를 들키지 마세요!</div>
                <div class="icon-circle">?</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {progress_width}%;"></div>
                </div>
            </div>
        """
    else:
        role_content = f"""
            <div class="role-card citizen">
                <div class="round-indicator">라운드 {game.current_round}/{game.total_rounds}</div>
                <div class="role-text">
                    <br>
                    당신은 시민입니다!
                </div>
                <div class="sub-text">주제: {game.chosen_topic}</div>
                <div class="sub-text">제시어: {st.session_state.secret_word}</div>
                <div class="sub-text">라이어를 찾으세요!</div>
                <div class="icon-circle">?</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {progress_width}%;"></div>
                </div>
            </div>
        """
    
    st.markdown(role_content, unsafe_allow_html=True)
    
    if st.button("다음 단계로"):
        st.session_state.game_phase = 'explanation'
        st.rerun()

# 설명 단계
elif st.session_state.game_phase == 'explanation':
    game = st.session_state.game
    display_game_info()
    current_player = st.session_state.players_order[st.session_state.current_player_idx]
    
    st.markdown("""
        <div class="explanation-card">
            <h2>설명 단계</h2>
            <p>각 플레이어는 제시어에 대해 한 문장씩 설명해주세요.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 메인 화면에 게임 정보 표시
    human_player = next(p for p in game.players if p.is_human)
    role_style = "liar" if human_player.is_liar else "citizen"
    st.markdown(f"""
        <div class="role-card {role_style}" style="margin-bottom: 20px;">
            <div class="player-name">{human_player.name}님의 게임 정보</div>
            <div class="sub-text">역할: {human_player.is_liar and '라이어' or '시민'}</div>
            <div class="sub-text">주제: {game.chosen_topic}</div>
            {'<div class="sub-text">제시어: ' + st.session_state.secret_word + '</div>' if not human_player.is_liar else ''}
        </div>
    """, unsafe_allow_html=True)
    
    # 이전 설명들 표시
    if st.session_state.descriptions:
        st.markdown("""
            <div style="margin: 20px 0;">
                <h3 style="color: white; margin-bottom: 15px;">지금까지의 설명:</h3>
        """, unsafe_allow_html=True)
        
        for name, desc in st.session_state.descriptions.items():
            st.markdown(f"""
                <div style="color: white; margin: 10px 0; line-height: 1.6;">
                    <span style="font-weight: bold;">{name}</span>: {desc}
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # 현재 플레이어 순서 표시
    st.markdown(f"""
        <div class="player-turn">
            <h3>{current_player.name}의 차례입니다</h3>
            <div class="icon-circle">?</div>
        </div>
    """, unsafe_allow_html=True)
    
    # 현재 플레이어의 설명 처리
    if current_player.is_human:
        if current_player.name not in st.session_state.descriptions:
            # 라이어일 때만 힌트 버튼 표시
            if current_player.is_liar and 'hint_shown' not in st.session_state:
                if st.button("힌트 받기"):
                    aggregated_comments = " ".join(st.session_state.descriptions.values())
                    predicted_words = game.predict_secret_word_from_comments(aggregated_comments)
                    top_5_words = list(predicted_words.keys())[:5]
                    st.session_state.liar_word_prediction = f"예측 단어: {', '.join(top_5_words)}"
                    st.session_state.hint_shown = True
                    st.rerun()
            
            # 힌트 표시 (라이어일 때만)
            if current_player.is_liar and 'hint_shown' in st.session_state and st.session_state.liar_word_prediction:
                st.markdown(f"""
                    <div class="hint-box">
                        <h4>🎯 힌트</h4>
                        <p>{st.session_state.liar_word_prediction}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # 설명 입력
            st.markdown("""
                <div style="margin: 20px 0;">
                    <h3 style="color: white;">당신의 설명을 입력하세요</h3>
                </div>
            """, unsafe_allow_html=True)
            explanation = st.text_input(
                "설명 입력",
                key="explanation_input",
                label_visibility="collapsed"
            )
            if st.button("제출하기"):
                if explanation:
                    st.session_state.descriptions[current_player.name] = explanation
                    st.session_state.current_player_idx += 1
                    if st.session_state.current_player_idx >= len(game.players):
                        st.session_state.game_phase = 'voting'
                    st.rerun()
    else:
        # AI 플레이어 설명 생성
        if current_player.name not in st.session_state.descriptions:
            aggregated_comments = " ".join(st.session_state.descriptions.values())
            if current_player.is_liar:
                explanation, _ = game.generate_ai_liar_description(aggregated_comments)
            else:
                explanation = game.generate_ai_truth_description(st.session_state.secret_word)
            st.session_state.descriptions[current_player.name] = explanation
        
        st.markdown(f"""
            <div style="margin: 20px 0; color: white;">
                <h3>AI의 설명</h3>
                <p style="margin-top: 10px;">{st.session_state.descriptions[current_player.name]}</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("다음 플레이어로"):
            st.session_state.current_player_idx += 1
            if st.session_state.current_player_idx >= len(game.players):
                st.session_state.game_phase = 'voting'
            st.rerun()

# 투표 단계
elif st.session_state.game_phase == 'voting':
    game = st.session_state.game
    display_game_info()
    
    st.markdown("""
        <div class="vote-card">
            <h2>투표 시간!</h2>
            <div class="icon-circle">⚖️</div>
            <p>모든 설명을 듣고 라이어를 찾아주세요.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 모든 설명 리뷰
    st.markdown("""
        <div class="explanation-card">
            <h3>플레이어들의 설명:</h3>
    """, unsafe_allow_html=True)
    
    for name, desc in st.session_state.descriptions.items():
        st.markdown(f"""
            <div class="description-box">
                <h4>{name}</h4>
                <p>{desc}</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 투표 섹션
    human_player = next(p for p in game.players if p.is_human)
    if human_player.name not in st.session_state.votes:
        st.markdown("""
            <div class="vote-card">
                <h3>투표하기</h3>
            </div>
        """, unsafe_allow_html=True)
        
        vote_options = [p.name for p in game.players if p != human_player]
        human_vote = st.selectbox("라이어라고 생각하는 플레이어를 선택하세요", vote_options)
        
        if st.button("투표 제출"):
            st.session_state.votes[human_player.name] = human_vote
            # AI 플레이어들의 투표
            for player in game.players:
                if not player.is_human:
                    vote = game.generate_ai_vote(player, st.session_state.descriptions)
                    st.session_state.votes[player.name] = vote
            st.session_state.game_phase = 'result'
            st.rerun()

# 결과 단계
elif st.session_state.game_phase == 'result':
    game = st.session_state.game
    display_game_info()
    
    # 투표 결과 집계
    vote_counts = {}
    for vote in st.session_state.votes.values():
        vote_counts[vote] = vote_counts.get(vote, 0) + 1
    
    st.markdown("""
        <div class="role-card">
            <h2>투표 결과</h2>
            <div class="icon-circle">📊</div>
        </div>
    """, unsafe_allow_html=True)
    
    # 투표 결과 표시
    for name, count in vote_counts.items():
        st.markdown(f"""
            <div class="vote-box">
                <h4>{name}</h4>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {(count/len(game.players))*100}%;"></div>
                </div>
                <p>{count}표</p>
            </div>
        """, unsafe_allow_html=True)
    
    # 점수 계산
    if 'points_calculated' not in st.session_state:
        
        highest_votes = max(vote_counts.values())
        top_candidates = [name for name, cnt in vote_counts.items() if cnt == highest_votes]
        original_scores = {player.name: player.score for player in game.players}
        
        # 라이어 공개
        st.markdown(f"""
            <div class="role-card liar">
                <h2>라이어 공개!</h2>
                <div class="icon-circle">🎭</div>
                <p>실제 라이어는 {game.liar.name}입니다!</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 라이어가 지목된 경우
        if game.liar.name in top_candidates:
            # 시민 승리
            st.markdown("""
                <div class="role-card citizen">
                    <h3>시민팀 승리!</h3>
                    <p>라이어가 지목되었습니다!</p>
                </div>
            """, unsafe_allow_html=True)
            
            # 시민들 점수 추가 (1번만 실행되도록)
            if not st.session_state.get('citizens_scored', False):
                for player in game.players:
                    if not player.is_liar:
                        player.score = original_scores[player.name] + 1
                        st.markdown(f"""
                            <div class="score-update">
                                <p>{player.name}이(가) 1점을 획득했습니다!</p>
                            </div>
                        """, unsafe_allow_html=True)
                st.session_state.citizens_scored = True
            
            # 라이어의 제시어 맞추기
            if game.liar.is_human:
                st.markdown("""
                    <div class="explanation-card">
                        <h3>마지막 기회!</h3>
                        <p>제시어를 맞추면 3점을 획득할 수 있습니다.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                liar_guess = st.text_input("제시어를 맞춰보세요")
                if st.button("정답 제출"):
                    if liar_guess.lower() == st.session_state.secret_word.lower():
                        if not st.session_state.get('liar_scored', False):
                            game.liar.score = original_scores[game.liar.name] + 3
                            st.markdown(f"""
                                <div class="role-card liar">
                                    <h3>대단해요!</h3>
                                    <p>{game.liar.name}이(가) 제시어를 맞추어 3점을 획득했습니다!</p>
                                </div>
                            """, unsafe_allow_html=True)
                            st.session_state.liar_scored = True
                    else:
                        st.markdown(f"""
                            <div class="explanation-card">
                                <p>{game.liar.name}이(가) 정답을 맞히지 못했습니다.</p>
                                <p>제시어는 '{st.session_state.secret_word}'였습니다!</p>
                            </div>
                        """, unsafe_allow_html=True)
                    st.session_state.points_calculated = True
                    
            else:
                # AI 라이어의 제시어 추측
                aggregated_comments = " ".join(st.session_state.descriptions.values())
                predicted_words = game.predict_secret_word_from_comments(aggregated_comments)
                liar_guess = list(predicted_words.keys())[0]
                
                st.markdown(f"""
                    <div class="explanation-card">
                        <h3>AI 라이어의 마지막 도전</h3>
                        <p>AI가 예측한 제시어: '{liar_guess}'</p>
                    </div>
                """, unsafe_allow_html=True)
                
                if liar_guess.lower() == st.session_state.secret_word.lower():
                    if not st.session_state.get('liar_scored', False):
                        game.liar.score = original_scores[game.liar.name] + 3
                        st.markdown(f"""
                            <div class="role-card liar">
                                <p>{game.liar.name}이(가) 제시어를 맞추어 3점을 획득했습니다!</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.session_state.liar_scored = True
                else:
                    st.markdown(f"""
                        <div class="explanation-card">
                            <p>{game.liar.name}이(가) 제시어를 맞추지 못했습니다.</p>
                            <p>제시어는 '{st.session_state.secret_word}'였습니다!</p>
                        </div>
                    """, unsafe_allow_html=True)
                st.session_state.points_calculated = True
                
        else:
            # 라이어 승리
            st.markdown(f"""
                <div class="role-card liar">
                    <h3>라이어 승리!</h3>
                    <p>라이어 {game.liar.name}이(가) 성공적으로 위장했습니다!</p>
                    <p>제시어는 '{st.session_state.secret_word}'였습니다!</p>
                </div>
            """, unsafe_allow_html=True)
            
            if not st.session_state.get('liar_scored', False):
                game.liar.score = original_scores[game.liar.name] + 1
                st.markdown(f"""
                    <div class="score-update">
                        <p>라이어({game.liar.name})가 1점을 획득했습니다!</p>
                    </div>
                """, unsafe_allow_html=True)
                st.session_state.liar_scored = True
            st.session_state.points_calculated = True

    # 다음 라운드 진행
    if 'points_calculated' in st.session_state:
        if st.button("다음 라운드로"):
            game.current_round += 1
            st.session_state.descriptions = {}
            st.session_state.votes = {}
            st.session_state.current_player_idx = 0
            st.session_state.round_data_initialized = False
            if 'points_calculated' in st.session_state:
                del st.session_state.points_calculated
            if 'hint_shown' in st.session_state:
                del st.session_state.hint_shown
            
            if game.current_round <= game.total_rounds:
                st.session_state.game_phase = 'role_reveal'
            else:
                st.session_state.game_phase = 'game_over'
            st.rerun()

# 게임 종료
elif st.session_state.game_phase == 'game_over':
    game = st.session_state.game
    
    # 게임 종료 타이틀
    st.markdown("""
        <div class="role-card" style="background: linear-gradient(135deg, #FFD700 0%, #FF8C00 100%);">
            <h2>게임 종료!</h2>
            <div class="icon-circle">🏆</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 100%;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 최종 점수 표시
    st.markdown("""
        <div class="explanation-card">
            <h3>🎯 최종 점수</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # 각 플레이어의 점수를 정렬하여 표시
    sorted_players = sorted(game.players, key=lambda x: x.score, reverse=True)
    
    # 점수 컨테이너 생성
    st.markdown("""
        <div style="display: flex; justify-content: center; gap: 20px; margin: 20px 0;">
    """, unsafe_allow_html=True)
    
    for i, player in enumerate(sorted_players):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "👏"
        st.markdown(f"""
            <div style="text-align: center; background: rgba(255,255,255,0.1); 
                        padding: 15px; border-radius: 15px; 
                        min-width: 150px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h4>{medal} {player.name}</h4>
                <p style="font-size: 1.5em; font-weight: bold; margin: 10px 0;">{player.score}점</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 승자 발표
    max_score = max(player.score for player in game.players)
    winners = [player.name for player in game.players if player.score == max_score]
    
    winner_announcement = f"""
        <div class="role-card" style="background: linear-gradient(135deg, #FFD700 0%, #FF8C00 100%); margin-top: 30px;">
            <h2>👑 최종 우승자</h2>
            <div style="font-size: 1.5em; margin: 20px 0;">
                {winners[0] if len(winners) == 1 else ', '.join(winners)}
            </div>
            <p>{'축하합니다!' if len(winners) == 1 else '공동 우승을 축하합니다!'}</p>
            <div class="icon-circle">✨</div>
        </div>
    """
    st.markdown(winner_announcement, unsafe_allow_html=True)
    
    # 새 게임 시작 버튼
    st.markdown("""
        <div class="explanation-card" style="text-align: center;">
            <h3>새로운 게임을 시작하시겠습니까?</h3>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("새 게임 시작"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()