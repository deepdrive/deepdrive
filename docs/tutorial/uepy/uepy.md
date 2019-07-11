# UnrealEnginePython Tutorial

The following tutorial demonstrates the power of UnrealEnginePython to manipulate objects in the running game with Python, i.e. `import unreal_engine`. For more tutorials on how to use UnrealEnginePython (UEPy) see [their docs](https://github.com/deepdrive/UnrealEnginePython/tree/master/tutorials).

Here, we'll show how to print to the Unreal Logs, get the ego vehicle (your car), and move the ego vehicle around.

Find your sim binaries by opening up `~/Deepdrive` and sorting by date 

![Find sim bin](/docs/tutorial/uepy/find-sim-bin.png)

Drill down to the Deepdrive binary

![Drill down](/docs/tutorial/uepy/sim-bin-drill-down.png)

Open a terminal, drag the file into the terminal, and press enter to open the sim in the terminal and see the logs.

![Terminal open sim](/docs/tutorial/uepy/terminal-open-sim.png)


Once the sim is open and you see the, press `M` to drive the car manually.

Within your binary folder, i.e. something like `deepdrive-sim-linux-3.0.20190528000032` open your `DeepDrive/Content/Scripts` in your favorite editor and create a new Python file named `move_car_tutorial.py` and enter:  

```python
print('hello world')
```

![Editor open](/docs/tutorial/uepy/editor-open.png)

Now hit the backtick `` ` `` key to open the Unreal Console and enter

```
py.exec move_car_tutorial.py
```

![Unreal Console Hello World](/docs/tutorial/uepy/unreal-console-hello-world.png)

You should then see "hello world" printed to the logs in the terminal:

![Terminal Hello World](/docs/tutorial/uepy/terminal-hello-world.png)

Now you know how to run Python within Unreal Engine. Let's do something more interesting!

Let's fly! Paste the following into `move_car_tutorial.py`


```python
import json
from api_methods import Api


api = Api()
ego = api.get_ego_agent()
location = ego.get_actor_location()
ego.set_actor_location(location.x, location.y, location.z + 1e4)  # +100m
```

Now open the Unreal Console again in the simulator with `` ` `` and hit the up arrow to rerun the previous command which should be 

```
py.exec move_car_tutorial.py
```

Weeeee, that was fun.

Often when developing a UEPy script, you won't know the exact name of the object you want to manipulate. Some general methods for introspecting the games internal state are demonstrated with the following script

```python
import json
from api_methods import Api, best_effort_serialize


api = Api()
ego = api.get_ego_agent()

print('Methods -------------------------------')
for m in dir(ego):
    print(m)

print('Properties --------------------------- ')
print(json.dumps(best_effort_serialize(ego, levels=5), indent=2))

```

[Example output](https://gist.githubusercontent.com/crizCraig/b9f9f86dc404a5658a85328e490d585e/raw/111e2e717d06ccd928683a782d7a70009a785a62/gistfile1.txt) 

Here you can see the wealth of functionality and information UnrealEnginePython provides. Imagine learning the game's state information with the [input remapping trick](https://arxiv.org/abs/1504.00702)! 
