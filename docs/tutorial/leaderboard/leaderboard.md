# Submitting to the leaderboard

## Overview 

The Deepdrive leaderboard uses [Botleague](https://github.com/botleague/botleague) to evaluate submissions. To place an agent on the leaderboards, you just need to submit a bot.json file via pull request which points to the docker tag for your bot.

We've provided an example project called [forward-agent](https://github.com/deepdrive/forward-agent/) that contains necessary code to get started.

Est. time (5-15 minutes depending on your internet cxn)

## Step 1: Clone the example repo

```
git clone https://github.com/deepdrive/forward-agent
```

## Step 2 (optional): Point to your docker image 

> If you want to just use the default bot, you can skip this step and go straight to:  [Step 4: Submit a pull request to Botleague](#step-4-submit-a-pull-request-to-botleague)

In the `Makefile` of the forward-agent repo, change the docker tag `TAG=deepdrive/forward-agent` to one that you have push access to, i.e. `TAG=yourdockerhubname/forward-agent`. 

## Step 3 (optional): If you performed step 2, build and push your bot 

Now build and push the docker image containing your the forward-agent bot to tag
you defined in the previous step with:

```
cd forward-agent
make && make push
```

## Step 4: Submit a pull request to Botleague

* Login to your GitHub account and fork the [botleague](https://github.com/botleague/botleague) repo with the fork button on the top right. Botleague is not a repo you _need_ to have locally, it merely functions as way to submit to the leaderboard with pull requests.

<hr>

![fork botleague](https://i.imgur.com/tgesEjc.jpg)

<hr>

* In your fork, create a file directly in GitHub using the `Create new file` button): 

<hr>

![create bot](https://i.imgur.com/NW1v9yt.jpg)

<hr>

Now we'll add a `bot.json` under `bots/YOUR-GITHUB-USERNAME/forward-agent`. GitHub will create the directories for you if you  paste `bots/<YOUR-GITHUB-NAME>/forward-agent/bot.json` replacing `YOUR-GITHUB-NAME` with your GitHub username.

<hr>

![paste bot name with directories](https://i.imgur.com/2ZRS6y3.png)

<hr>

Now add the following JSON to the file and click <kbd>Commit new file</kdb>

> Note: if did step 2 and 3, replace `crizcraig` with your DockerHub or other registry name in the JSON below

### JSON for bot
```
{ 
  "docker_tag": "crizcraig/forward-agent",
  "problems": ["deepdrive/unprotected_left"] 
}
```

Submit a pull request to the main botleague repo from your fork

<hr>

![paste bot name with directories](https://i.imgur.com/DsFddJQ.jpgg)

<hr>

* The pull request will be automatically merged after about 10 minutes once the evaluation is complete
* Click the View Details=>Details button on the pull request to see your bot's results
* Finally, check the [leaderboards](https://deepdrive.voyage.auto/leaderboard) to see your bot's score and video ranked among the other bots!

## Next steps

### Local development of your bot

To test your bot locally, it's ideal to run the sim and agent on your local machine as in our [examples](https://docs.deepdrive.io/#examples). You can see what your bot scores locally by passing the [same parameters](https://github.com/deepdrive/deepdrive/blob/f93e1091cdd9e393fd5516eedbf85e19e380773c/botleague/problems/unprotected_left/run.sh#L10) to `main.py` as we do on the evaluation server excluding the `--server` parameter.

Next you can run the sim in server mode locally with those [same parameters](https://github.com/deepdrive/deepdrive/blob/f93e1091cdd9e393fd5516eedbf85e19e380773c/botleague/problems/unprotected_left/run.sh#L10) again, but this time, keeping `--server` in the params passed to `main.py`.

Finally, make sure your bot runs as a docker container against the official scenario container. For the case of `unprotected_left`, for example, the docker image would be `deepdriveio/deepdrive:problem_unprotected_left`. You can see how our scenario problem images are built [here](https://github.com/deepdrive/deepdrive/tree/e565f52794c1d18904f1b2fc7c79a05e8629ed46/botleague/problems).

Then to build your bot container, checkout how our baseline agent bot containers are built [here](https://github.com/deepdrive/deepdrive/tree/e565f52794c1d18904f1b2fc7c79a05e8629ed46/botleague/bots).

:shipit: Then to submit your bot, create a pull request as we did above, pointing to your bot's docker image. If the image is the same, you can just add whitespace, or change some comment text to trigger the pull request.




