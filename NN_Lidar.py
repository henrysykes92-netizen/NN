from Lidar import *

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
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )
        '''
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
        '''
    def forward(self, x):
        return self.model(x)

def flatten(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

def get_state(board, player):
    s = [
        player.hunger,
        round(board.player_x, 1),
        round(board.player_y, 1),
        round(board.obj_x, 1),
        round(board.obj_y, 1)
    ]
    # Flatten lidar.area and lidar.objects and append to state
    s += list(flatten(lidar.area))
    s += list(flatten(lidar.objects))
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
    lidar = Lidar()

    s = [
        player.hunger,
        round(board.player_x, 1),
        round(board.player_y, 1),
        round(board.obj_x, 1),
        round(board.obj_y, 1)
    ]
    s += list(flatten(lidar.area))
    s += list(flatten(lidar.objects))
    input_size = len(s)

    output_size = 3
    hidden_size = 128 #360
    gamma = 0.95
    epsilon = 1
    epsilon_decay = 0.99992  # 1 + math.log(0.9999, board.hunger * board.fps)
    epsilon_min = 0.05
    learning_rate = 1e-4  # 1e-4

    model = QNetwork(input_size, hidden_size, output_size)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    buffer = ReplayBuffer(capacity=10000)
    batch_size = 64

    # prev_score = 0

    loop = 0
    rend_loop = 1 #5000
    non_r = 500

    with open("Lidar_Data\\Values.txt", "a", newline="") as f:
        f.write(f'Gamma: {gamma}')
        f.write(f'\tEpsilon decay: {epsilon_decay}')
        f.write(f'\tLearning rate: {learning_rate}')
        f.write(f'\nLoop\tScore\t\tFound\tTicks\tEpsilon')

    # Training Loop
    loop = 0

    #pygame.init()
    #clock = pygame.time.Clock()

    while True:
        pygame.init()
        clock = pygame.time.Clock()
        loop += 1
        ticks = 0
        print(f'Loop: {loop}')
        run = Run()
        if loop % rend_loop == 0:
            lidar.render_init()

        state = get_state(board, player).unsqueeze(0)
        action = select_action(state, model, epsilon, output_size)
        prev_score = player.score

        # -- Game Loop --
        found = 0
        while player.alive:
            ticks += 1

            if action == 0:
                # player.score += 1/(board.hunger * board.fps)
                speed = board.max_speed
            elif action == 1:
                speed = 0
                board.player_angle += board.max_rotate
            else:
                speed = 0
                board.player_angle -= board.max_rotate

            board.player_x += math.cos(math.radians(board.player_angle)) * board.max_speed  # speed
            board.player_y += math.sin(math.radians(board.player_angle)) * board.max_speed  # speed

            #print(ticks)
            '''#for event in pygame.event.get():
             #   if event.type == pygame.QUIT:
              #      break
            if player.found != found:
                found = player.found
                lidar.objects = [None] * int(lidar.memory / 40)

            keys = pygame.key.get_pressed()
            board.player_angle += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * board.max_rotate
            speed = keys[pygame.K_UP] * board.max_speed
            # speed = ((keys[pygame.K_UP]*player.forward) - keys[pygame.K_DOWN]*player.backward) * board.max_speed
            # speed = ((keys[pygame.K_UP]) - keys[pygame.K_DOWN]) * board.max_speed

            board.player_x += math.cos(math.radians(board.player_angle)) * speed  # * player.move_x
            board.player_y += math.sin(math.radians(board.player_angle)) * speed  # * player.move_y
            '''

            run.step()
            #if loop % rend_loop == 0:
             #   run.render_step()

            for ray in rays:
                #print(ray.object)
                if ray.object == (125, 125, 0):
                    #print((ray.end_x, ray.end_y))
                    lidar.bounce(ray.end_x, ray.end_y, False)
                if ray.object == (0, 0, 255):
                    lidar.bounce(ray.end_x, ray.end_y, True)

            if loop % rend_loop == 0:
                lidar.render()
                pygame.display.flip()
                clock.tick(board.fps)

            next_state = get_state(board, player).unsqueeze(0)
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
                with open(str(f'Lidar_Data\\Loop_{loop}_Training_Data.txt'), "a", newline="") as f:
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
            torch.save(model.state_dict(), f"Lidar_Data\\Loop_{loop}_Model.pkl")
            images = []
            # for i in range(frame_count):
            #   filename = f"{frame_dir}/frame_{i:04d}.png"
            #  images.append(iio.v3.imread(filename))
            # imageio.mimsave(f"Data/Loop_{loop}_render.gif", images, fps=board.fps)

        with open("Lidar_Data\\Values.txt", "a", newline="") as f:
            f.write(f'\n{loop}\t{round(player.score, 6):.6f}\t{player.found}\t{ticks}\t{epsilon}')

            #if player.alive == False:
             #   print(f'dead\t{player.score}')
              #  break

        if loop % rend_loop == 0:
            pygame.quit()
