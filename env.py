import numpy as np
import matplotlib.pyplot as plt
import time

class ActiveSlamEnv:
    """
    Active SLAM을 위한 2D 그리드 월드 환경
    - true_map: 환경이 아는 실제 정답 지도 (1:벽, 0:길)
    - belief_map: 에이전트가 탐색하며 구축하는 지도 (-1:미지, 0:길, 1:벽)
    """

    # 지도 범례
    MAP_LEGEND = {'UNKNOWN': -1, 'EMPTY': 0, 'WALL': 1}
    # 렌더링 시 에이전트 위치 표시
    AGENT_VIEW = 5 

    def __init__(self, map_file_path):
        # 1. 정답 지도 (True Map) 로드
        self.true_map = np.loadtxt(map_file_path, dtype=int)
        self.height, self.width = self.true_map.shape

        # 2. 에이전트 방향 (0:동, 1:북, 2:서, 3:남) 및 이동 벡터
        self.agent_dir = 0
        self.directions = [
            (0, 1),  # 0: 동
            (-1, 0), # 1: 북
            (0, -1), # 2: 서
            (1, 0)   # 3: 남
        ]

        # 3. Action 정의 (0:직진, 1:좌회전, 2:우회전)
        self.action_space = [0, 1, 2]

        # 4. 시각화 설정
        plt.ion() # Matplotlib 대화형 모드 켜기
        self.fig, self.ax = plt.subplots()

        # 5. 환경 초기화
        self.reset()

    def reset(self):
        """환경을 초기화합니다."""
        # 1. 신념 지도(Belief Map)를 모두 '미지'(-1)로 초기화
        self.belief_map = np.full((self.height, self.width), self.MAP_LEGEND['UNKNOWN'], dtype=int)

        # 2. 에이전트의 시작 위치 탐색 (가장 처음 나오는 '길')
        start_pos_arr = np.argwhere(self.true_map == self.MAP_LEGEND['EMPTY'])
        self.agent_pos = tuple(start_pos_arr[0])

        # 3. 시작 위치는 '길'임을 알게 됨
        self.belief_map[self.agent_pos] = self.MAP_LEGEND['EMPTY']
        
        # 4. 초기 관측 수행 (LIDAR 스캔)
        self._update_belief_map_with_lidar()

        # 5. 초기 관측(State) 반환
        return self._get_observation()

    def _get_observation(self):
        """현재 State (관측)를 반환합니다."""
        # DQN의 입력으로 사용하기 위해 belief_map을 1차원으로 펼쳐서 반환
        # (더 정교한 State 설계가 가능함)
        return self.belief_map.flatten()

    def _update_belief_map_with_lidar(self):
        """
        현재 위치에서 8방향 레이캐스팅(LIDAR)을 시뮬레이션합니다.
        true_map을 참조하여 belief_map을 업데이트합니다.
        """
        newly_discovered_cells = 0
        x, y = self.agent_pos

        # 8방향 (대각선 포함)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                # (x, y)에서 (dx, dy) 방향으로 레이 발사
                ray_x, ray_y = x, y
                while True:
                    ray_x, ray_y = ray_x + dx, ray_y + dy

                    # 1. 맵 경계를 벗어나는지 확인
                    if not (0 <= ray_x < self.height and 0 <= ray_y < self.width):
                        break # 레이가 맵 밖으로 나감

                    # 2. true_map에서 현재 레이 위치의 값 확인
                    true_val = self.true_map[ray_x, ray_y]

                    # 3. belief_map 업데이트
                    if self.belief_map[ray_x, ray_y] == self.MAP_LEGEND['UNKNOWN']:
                        self.belief_map[ray_x, ray_y] = true_val
                        newly_discovered_cells += 1
                    
                    # 4. 벽을 만나면 레이가 멈춤
                    if true_val == self.MAP_LEGEND['WALL']:
                        break
                        
        return newly_discovered_cells

    def step(self, action):
        """
        Action을 수행하고 (next_observation, reward, done)을 반환합니다.
        """
        # 1. Action에 따른 에이전트 상태 변경
        if action == 1: # 좌회전 (0->1, 1->2, 2->3, 3->0)
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == 2: # 우회전 (0->3, 3->2, 2->1, 1->0)
            self.agent_dir = (self.agent_dir - 1 + 4) % 4
        
        # 2. 이동할 다음 위치 계산 (Action 0: 직진)
        is_collision = False
        new_pos = list(self.agent_pos)
        
        if action == 0: # 직진
            dr, dc = self.directions[self.agent_dir]
            new_pos[0] += dr
            new_pos[1] += dc

        # 3. 충돌 검사 (true_map 기준)
        if self.true_map[new_pos[0], new_pos[1]] == self.MAP_LEGEND['WALL']:
            is_collision = True
        else:
            # 충돌하지 않았으면 위치 업데이트
            self.agent_pos = tuple(new_pos)

        # 4. 새 위치에서 LIDAR 스캔 및 belief_map 업데이트
        newly_discovered_cells = self._update_belief_map_with_lidar()

        # 5. 보상(Reward) 계산
        reward = 0
        if is_collision:
            reward -= 5.0 # 충돌 페널티
        
        # 새로 발견한 셀 개수만큼 보상 (탐험 장려)
        reward += newly_discovered_cells * 1.0 
        
        # 매 스텝마다 작은 페널티 (효율성 장려)
        reward -= 0.1 

        # 6. 종료 조건 (예: 95% 이상 탐색 완료)
        coverage = np.count_nonzero(self.belief_map != self.MAP_LEGEND['UNKNOWN']) / self.belief_map.size
        done = coverage > 0.95
        
        # 7. 다음 상태(Observation) 반환
        next_observation = self._get_observation()
        
        return next_observation, reward, done

    def render(self):
        """현재 belief_map과 에이전트 위치를 시각화합니다."""
        self.ax.clear()
        
        # belief_map을 복사하여 렌더링용 맵 생성
        display_map = self.belief_map.astype(float) # float로 변경
        
        # 에이전트 위치 표시
        display_map[self.agent_pos] = self.AGENT_VIEW
        
        # '미지'(-1)는 검은색, '길'(0)은 흰색, '벽'(1)은 회색, '에이전트'(5)는 빨간색
        self.ax.imshow(display_map, cmap='gray_r', vmin=-1, vmax=5)
        
        # 탐색률(Coverage) 표시
        coverage = np.count_nonzero(self.belief_map != self.MAP_LEGEND['UNKNOWN']) / self.belief_map.size
        self.ax.set_title(f"Active SLAM Belief Map (Coverage: {coverage:.2%})")
        
        plt.draw()
        plt.pause(0.01) # 잠시 멈춰서 그림을 업데이트