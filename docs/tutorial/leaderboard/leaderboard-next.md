# Submitting to the Deepdrive leaderboard

## Overview 

First-off we'll get you on the leaderboard with the default bot, so you can get familiar with the submission process. 
The Deepdrive leaderboard uses [Botleague](https://github.com/botleague/botleague) to evaluate submissions.
To place an agent on the leaderboards, you just need to submit a bot.json file to the league repo via pull request.

Est. time (5-15 minutes depending on your internet connection)

## Step 1: Fork botleague

Login to your GitHub account and fork the [botleague](https://github.com/botleague/botleague) repo with the fork button on the top right.


![fork botleague](https://i.imgur.com/tgesEjc.jpg)

## Step 2: Clone your fork

```
git clone https://github.com/<YOUR-GITHUB-NAME>/botleague
cd botleague
```

## Step 3: Create a bot.json

Create a `bots/<YOUR-GITHUB-NAME>/forward-agent/bot.json` in your fork with the following JSON.

```
{ 
  "docker_tag": "crizcraig/forward-agent",
  "problems": ["deepdrive/unprotected_left"] 
}
```

>NOTE: Here `crizcraig/forward-agent` is the default docker image for the forward-agent bot. Later on, when you modify your bot, you will replace this docker tag with a repo you have push access to.

## Step 3: Commit and push your bot.json

```
git commit -am 'forward-agent'
git push origin master
```

## Step 4: Click Pull Request on your repo's page

![click pull request](https://i.imgur.com/DsFddJQ.jpg)


## Step 4: Create your pull request

![create the pull request yay](https://i.imgur.com/CW77bha.jpg)


## Step 4: Confirm that botleague has started your evaluation

Your pull request status should update to something similar to the following

![pull request status update](https://i.imgur.com/bimSaQW.png)

## Step 5: Wait for your bots evaluation to complete

Grab a coffee! This will take a 5-10 minutes.

## Step 6: Verify your pull request is merged

Once your evaluation is complete, it will be automatically merged as displayed in the image below. You may need to refresh the page if you don't see this after 10 minutes.

![click ](https://i.imgur.com/6nffqfl.jpg)

## Step 7: Go to the leaderbaord!

Finally, check the [leaderboards](https://deepdrive.voyage.auto/leaderboard) to see your bot's score and video ranked among the others.

## Next steps

### Local development of your bot

To test your bot locally, it's ideal to run the sim and agent on your local machine as in our [examples](https://docs.deepdrive.io/#examples). You can see what your bot scores locally by passing the [same parameters](https://github.com/deepdrive/deepdrive/blob/f93e1091cdd9e393fd5516eedbf85e19e380773c/botleague/problems/unprotected_left/run.sh#L10) to `main.py` as we do on the evaluation server excluding the `--server` parameter.

Next you can run the sim in server mode locally with those [same parameters](https://github.com/deepdrive/deepdrive/blob/f93e1091cdd9e393fd5516eedbf85e19e380773c/botleague/problems/unprotected_left/run.sh#L10) again, but this time, keeping `--server` in the params passed to `main.py`.

Now make sure your bot runs as a docker container against the official scenario container. For the case of `unprotected_left`, for example, the docker image would be `deepdriveio/deepdrive:problem_unprotected_left`. You can see how our scenario problem images are built and run [here](https://github.com/deepdrive/deepdrive/tree/e565f52794c1d18904f1b2fc7c79a05e8629ed46/botleague/problems).

Then to build your bot container, refer to how our baseline agent bot containers are built [here](https://github.com/deepdrive/deepdrive/tree/e565f52794c1d18904f1b2fc7c79a05e8629ed46/botleague/bots).

Finally, to submit your bot, create a pull request as we did above, pointing to your bot's docker image. If the image is the same, you can just add whitespace, or change some comment text to allow for the pull request.
