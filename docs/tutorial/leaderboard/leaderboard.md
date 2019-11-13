# Submitting to the leaderboard

### Overview 

The Deepdrive leaderboard uses [Botleague](https://github.com/botleague/botleague) to evaluate submissions. To place an agent on the leaderboards, you just need to submit a bot.json file via pull request which points to the docker tag for your bot.

We've provided an example project called [forward-agent](https://github.com/deepdrive/forward-agent/) that contains necessary code to get started.

Est. time (5-15 minutes depending on your internet cxn)

### Step 1: Clone the example repo

```
git clone https://github.com/deepdrive/forward-agent
```

### Step 2: Point to your docker image (optional)

> If you want to just use the default bot, you can skip this step and go straight to:  [Step 4: Submit a pull request to Botleague](#step-4-submit-a-pull-request-to-botleague)

In the `Makefile` of the forward-agent repo, change the docker tag `TAG=deepdrive/forward-agent` to one that you have push access to, i.e. `TAG=yourdockerhubname/forward-agent`. 

### Step 3: Build and push your bot (optional)

Now build and push the docker image containing your the forward-agent bot to tag
you defined in the previous step with:

```
cd forward-agent
make && make push
```

### Step 4: Submit a pull request to Botleague

* Login to your GitHub account and fork the [botleague](https://github.com/botleague/botleague) repo with the fork button on the top right
* In your fork, create `bots/yourgithubname/forward-agent/bot.json` and paste the JSON below replacing `yourdockerhubname` with whatever you chose in step 2. You can create files directly in GitHub using the `Create new file` button): 

> Note: if did step 2 and 3, replace `crizcraig` with your DockerHub or other registry name.

#### JSON for bot
```
{ 
  "docker_tag": "crizcraig/forward-agent",
  "problems": ["deepdrive/unprotected_left"] 
}
```

* Submit a pull request to the main botleague repo from your fork
* The pull request will be automatically merged after about 2 minutes once the evaluation is complete
* Click the View Details button on the pull request to see your bot's results
* Finally, check the [leaderboards](https://deepdrive.voyage.auto/leaderboard) to see your bot's score and video ranked among the other bots!

TODO: Testing your agent locally by running deepdrive main.py --server ...  
