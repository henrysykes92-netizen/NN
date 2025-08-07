from Environment import *

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(torch.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        '''self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )'''

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    def forward(self, x):
        return self.model(x)

def get_state(board, player, rays):
    s = [round(board.player_x, 1), round(board.player_y, 1), round(board.obj_x, 1), round(board.obj_y, 1), player.hunger]
    for ray in rays:
        try:
            s.append(round(ray.distance, 1))
        except:
            s.append(0)
        s.append(1.0 if ray.object == (0, 0, 255) else 0.0)
    return torch.tensor(s, dtype=torch.float32)
def select_action(state, model, epsilon, output_size):
    if random.random() < epsilon:
        return random.randint(0, output_size - 1)
    with torch.no_grad():
        q_values = model(state)
        return torch.argmax(q_values).item()
def non_r_action(state, model):
    with torch.no_grad():
        q_values = model(state)
        return torch.argmax(q_values).item()
def sarsa_update(model, optimiser, criterion, state, action, reward, next_state, next_action, gamma):
    model.train()
    q_values = model(state)
    next_q_values = model(next_state)
    target = q_values.clone().detach()
    target[0, action] = reward + gamma * next_q_values[0, next_action]
    loss = criterion(q_values, target)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()


if __name__ == '__main__':
    input_size = len(rays)*2 + 5
    output_size = 3
    hidden_size = 128
    gamma = 0.9
    epsilon = 1
    epsilon_decay = 0.99992 # 1 + math.log(0.9999, board.hunger * board.fps)
    epsilon_min = 0.05
    learning_rate = 1e-4 #1e-4

    model = QNetwork(input_size, hidden_size, output_size)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    buffer = ReplayBuffer(capacity=10000)
    batch_size = 64

    #prev_score = 0

    clock = pygame.time.Clock()
    loop = 0
    rend_loop = 0.1 #5000
    non_r = 500

    with open("Data\\Values.txt", "a", newline="") as f:
        f.write(f'Gamma: {gamma}')
        f.write(f'\tEpsilon decay: {epsilon_decay}')
        f.write(f'\tLearning rate: {learning_rate}')
        f.write(f'\nLoop\tScore\t\tFound\tTicks\tEpsilon')

    while True:
        loop += 1
        ticks = 0
        print(f'Loop: {loop}')
        board.__init__()
        player.__init__(x=board.player_x, y=board.player_y, angle=board.player_angle)

        offsets = []
        for i in np.arange(-board.fov / 2, board.fov / 2 + 1, board.fov / board.no_ofrays):
            offsets.append(i)
            rays[int((i-1)*2/board.fov)].__init__(start_x=board.player_x, start_y=board.player_y, start_angle=board.player_angle, offset=i)
        #if 0 not in offsets:
         #   rays[-1].__init__(start_x=board.player_x, start_y=board.player_y, start_angle=board.player_angle, offset=0)

        run = Run()
        if loop % rend_loop == 0:
            run.render_init()

            #frame_dir = f"Data/Loop_{loop}_frames"
            #os.makedirs(frame_dir, exist_ok=True)
            #frame_count = 0

            with open(str(f'Data\\Loop_{loop}_Training_Data.txt'), "w", newline="") as f:
                for row in board.board:
                    f.write(str(row)+'\n')
                header = ['p_x', '\tp_y', '\to_x', '\to_y', '\thunger'] + [f'\tray_{i}_values' for i in range(len(rays))] + ['\taction']
                for h in header:
                    f.write(str(h))
                f.write('\n')

        state = get_state(board, player, rays).unsqueeze(0)
        action = select_action(state, model, epsilon, output_size)
        prev_score = player.score
        #speed = board.max_speed
        while True: #player.alive:
            ticks += 1

            if action == 0:
                #player.score += 1/(board.hunger * board.fps)
                speed = board.max_speed
            elif action == 1:
                speed = 0
                board.player_angle += board.max_rotate
            else:
                speed = 0
                board.player_angle -= board.max_rotate

            board.player_x += math.cos(math.radians(board.player_angle)) * board.max_speed#speed
            board.player_y += math.sin(math.radians(board.player_angle)) * board.max_speed#speed

            run.step()
            if loop % rend_loop == 0:
                run.render_step()
                pygame.display.update()
                clock.tick(board.fps)
                #pygame.image.save(pygame.display.get_surface(), f"{frame_dir}/frame_{frame_count:04d}.png")
                #frame_count += 1

            next_state = get_state(board, player, rays).unsqueeze(0)
            if player.found == 10:
                player.score = 25
                player.alive = False
            reward = player.score - prev_score
            done = not player.alive

            buffer.push(state.squeeze(0), torch.tensor([action]), torch.tensor([reward], dtype=torch.float32),
                        next_state.squeeze(0), torch.tensor([done], dtype=torch.float32))

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                q_values = model(states).gather(1, actions)
                next_q_values = model(next_states).max(1)[0].unsqueeze(1)
                expected_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = criterion(q_values, expected_q_values.detach())
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            state = next_state
            if loop % non_r == 0:
                action = non_r_action(state, model)
            else:
                action = select_action(state, model, epsilon, output_size)
            prev_score = player.score

            '''if epsilon > epsilon_min:
                epsilon *= epsilon_decay'''

            if loop % rend_loop == 0:
                with open(str(f'Data\\Loop_{loop}_Training_Data.txt'), "a", newline="") as f:
                    for s in state.flatten().tolist():
                        f.write(f'{s:.3f}')
                        f.write('\t')
                    f.write('\n')

            if not player.alive:
                break

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if loop % rend_loop == 0:
            pygame.quit()
            torch.save(model.state_dict(), f"Data\\Loop_{loop}_Model.pkl")
            images = []
            #for i in range(frame_count):
             #   filename = f"{frame_dir}/frame_{i:04d}.png"
              #  images.append(iio.v3.imread(filename))
            #imageio.mimsave(f"Data/Loop_{loop}_render.gif", images, fps=board.fps)

        with open("Data\\Values.txt", "a", newline="") as f:
            f.write(f'\n{loop}\t{round(player.score,6):.6f}\t{player.found}\t{ticks}\t{epsilon}')
