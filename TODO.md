- [x] Change random action to be controlled from Python by just sending a random action - just pull in Maik's and use new send action to do it
- [x] Implement a reset - right now pressing R respawns, but vehicle motion is not reset.
- [x] Put a delay in between actions so as to maintain a constant frame recording and action rate
- [x] Store data outside source dir to avoid indexing issues
- [x] Start Unreal via Python
- [x] Packaged version on windows not returning images / depth correctly (small values and zeroes, thinking it's a shared mem naming thing)
- [x] implement tryConnect in Linux
- [x] Provide steering, throttle, brake output from game and should_game_drive
- [x] Improve fps
- [ ] Verify shared mem version is in sync at runtime
- [ ] Change handbrake from boolean to float, make throttle forward only, and readd brake as brake/reverse (brake, forward throttle, and handbrake can be on at the same time)to match value in unreal
- [x] Make day night change faster
- [ ] Fix the dimension output
- [ ] Docs explaining the structure, singletons, proxies, registration, shared mem, workers, etc...
- [ ] Move Car.cpp / SplineTrajectory.h / and any other blueprints / code we created into the plugin
- [ ] Explain this ```int32 MaxSharedMemSize = 3 * (8 * 1024 * 1024 + 10240) + 10240;``` in plugin
- [ ] Allow passing resolution and number of cameras as command line args
- [ ] Sync PyCaptureSnapshotObject docs with https://docs.google.com/spreadsheets/d/1-z7fb7YtkSYD9Wda1LB358oZ5oHkP6F5dbpjOeskx5k/edit#gid=0
- [z] Change pause menu credits, remove soul dance party
- [ ] Remove all the SDP header comments in c++
- [x] Change SDP license plate
- [ ] Finish this comment in the plugin "// actually a bool but its better to repre"
- [ ] Remove GrabbingCameraActor
- [x] Establish connection handshake to establish size and name of shared memory (naming will allow multiple game instances to run and sizing allows variable camera configs to work more efficiently since we just allocate the max size now)
- [ ] Make shared memory name random to ease pain of changing it during dev

we have proxies because we couldn't have a singleton actor

reward
- distance travelled to destination along path
- lap number - if distance travelled closed to lap length and current distance close to zero, then achieved lap, continue to add distance
- comfort via accumulated acceleration / angular acceleration
- distance from center
- was a collision