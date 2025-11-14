import numpy as np
from env import ActiveSlamEnv # 위에서 만든 환경 파일 import
import time

# --- 하이퍼파라미터 ---
MAP_FILE = 'maze.txt'
TOTAL_EPISODES = 10     # 총 몇 번의 탐험을 시도할지
MAX_STEPS_PER_EPISODE = 500 # 한 번의 탐험에서 최대 몇 걸음까지 걸을지

# 1. 환경 생성
env = ActiveSlamEnv(map_file_path=MAP_FILE)

print("--- Active SLAM (Random Agent) 시작 ---")
print(f"환경: {MAP_FILE}")
print(f"Action Space: {env.action_space} (0:직진, 1:좌, 2:우)")
print(f"Observation Space (State): {env.belief_map.shape} (1D로 {env.belief_map.size}개)")

# 2. 총 에피소드만큼 반복
for episode in range(TOTAL_EPISODES):
    
    # 3. 환경 초기화
    obs = env.reset()
    total_reward = 0
    
    print(f"\n--- Episode {episode + 1}/{TOTAL_EPISODES} 시작 ---")

    # 4. 최대 스텝만큼 반복
    for step in range(MAX_STEPS_PER_EPISODE):
        
        # 5. "Random Algorithm" (무작위로 행동 선택)
        action = np.random.choice(env.action_space)

        # 6. 환경에 Action을 전달하고 결과(S', R, Done)를 받음
        next_obs, reward, done = env.step(action)

        total_reward += reward
        
        # 7. 환경 시각화
        env.render()

        # 8. 에피소드 종료 조건 확인
        if done:
            print(f"에피소드 {episode + 1} 종료: 95% 이상 탐색 완료! ( {step + 1} steps, Total Reward: {total_reward:.2f} )")
            break
        
        if step == MAX_STEPS_PER_EPISODE - 1:
            print(f"에피소드 {episode + 1} 종료: 최대 스텝 도달. ( {step + 1} steps, Total Reward: {total_reward:.2f} )")

print("\n--- 모든 시뮬레이션 종료 ---")

# 시각화 창 닫기
plt.ioff()
plt.show()