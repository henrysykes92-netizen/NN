from Environment import *

width, height = 400, 400
map_size = 8
tile_size = width / map_size
view_range = width / 2.5
fov = 60 # 90
no_ofrays = 4
max_speed = 5
max_rotate = 10# math.pi
fps = 12

player_radius = tile_size * 0.2
object_radius = tile_size * 0.4

pygame.init()
board_surf = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

'''
with open('Data\\Loop_4000_Training_Data.txt', 'r') as f:
    vals = f.readlines()
    data = []

    for row in range(map_size):
        #print(vals[row].split(','))
        line = vals[row].strip()
        parsed = ast.literal_eval(line)
        print(parsed)
        for col in range(map_size):
            if parsed[col] == '#':
                rect = (col * tile_size, row * tile_size, tile_size, tile_size)
                pygame.draw.rect(board_surf, (125, 125, 125), rect)

    data.append(list(vals[map_size].split()))
    for i in range((map_size + 1), len(vals)):
        player_x = float(list(vals[i].split())[1])
        player_y = float(list(vals[i].split())[1])
        obj_x = float(list(vals[i].split())[2])
        obj_y = float(list(vals[i].split())[3])

        pygame.draw.circle(board_surf, (0, 125, 0), (player_x, player_y), radius=player_radius)
        pygame.draw.circle(board_surf, (0, 0, 125), (obj_x, obj_y), radius=object_radius)

        pygame.display.update()
        clock.tick(fps)
'''

avg = 250
qv = 500
loop = []
loop_avg = []
loop_qv = []
score = []
score_avg = []
score_qv = []
found = []
found_avg = []
found_qv = []
ticks = []
ticks_avg = []
ticks_qv = []
epsilon = []

with open("Data\\Values.txt", "r") as f:
    data = f.readlines()
    for i in range(2, len(data)-1):
        vals = list(data[i].split())
        loop.append(int(vals[0]))
        score.append(float(vals[1]))
        found.append(float(vals[2]))
        ticks.append(float(vals[3]))
        epsilon.append(float(vals[4]))

for i in range(0,len(loop),int(avg/10)):
    loop_avg.append(i)
    try:
        avg_s = statistics.mean(score[i - avg:i + avg])
        avg_f = statistics.mean(found[i - avg:i + avg])
        avg_t = statistics.mean(ticks[i - avg:i + avg])
    except:
        try:
            avg_s = statistics.mean(score[0:i + avg])
            avg_f = statistics.mean(found[0:i + avg])
            avg_t = statistics.mean(ticks[0:i + avg])
        except:
            avg_s = statistics.mean(score[i - avg:-1])
            avg_f = statistics.mean(found[i - avg:-1])
            avg_t = statistics.mean(ticks[i - avg:-1])
    score_avg.append(avg_s)
    found_avg.append(avg_f)
    ticks_avg.append(avg_t)

for i in range(qv,len(loop),qv):
    loop_qv.append(loop[i-1])
    score_qv.append(score[i-1])
    found_qv.append(found[i-1])
    ticks_qv.append(ticks[i-1])

plt.figure()
plt.scatter(loop, score, color='green', s=0.1)
plt.plot(loop_avg, score_avg, color='orange')
plt.plot(loop_qv, score_qv, color='blue')
plt.title("Score Values")
plt.xlabel("Loop")
plt.ylabel("Score")
plt.grid(True)
plt.show()

plt.figure()
plt.scatter(loop, found, color='green', s=0.1)
plt.plot(loop_avg, found_avg, color='orange')
plt.plot(loop_qv, found_qv, color='blue')
plt.title("Found Values")
plt.xlabel("Loop")
plt.ylabel("Found")
plt.grid(True)
plt.show()

plt.figure()
plt.scatter(loop, ticks, color='green', s=0.1)
plt.plot(loop_avg, ticks_avg, color='orange')
plt.plot(loop_qv, ticks_qv, color='blue')
plt.title("Tick Values")
plt.xlabel("Loop")
plt.ylabel("Ticks")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(loop, epsilon, color='red')
plt.title("Epsilon Values")
plt.xlabel("Loop")
plt.ylabel("Epsilon")
plt.grid(True)
plt.show()


'''
# Enable interactive mode
plt.ion()

# Path to the saved model file
model_path = 'Data/Loop_800_Model.pkl'

# Load the model state dictionary
try:
    state_dict = torch.load(model_path)

    # Create histograms for weights and biases
    for param_name, param_tensor in state_dict.items():
        plt.figure(figsize=(80, 40))
        plt.hist(param_tensor.cpu().numpy().flatten(), bins=50, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {param_name}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        input("Press Enter to continue to the next plot...")

except FileNotFoundError:
    print(f"Model file not found at: {model_path}")
except Exception as e:
    print(f"An error occurred: {e}")
'''