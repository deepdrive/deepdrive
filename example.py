import sim


def main():
    env = sim.start(

        # map can be canyons, kevindale, kevindale_bare, or jamestown
        map='kevindale_bare',

        # scenario can be 0 => 5 - Scenario descriptions:
        # https://gist.github.com/crizCraig/855a5cc4bc96fc2765cb9bf5d6f953b4
        scenario_index=1,
        
        client_main_args = {'opengl', 3}
    )
    forward = sim.action(throttle=0.75, steering=0, brake=0)
    done = False
    while not done:
        observation, reward, done, info = env.step(forward)
    env.close()

if __name__ == '__main__':
    main()
