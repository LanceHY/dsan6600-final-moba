# dsan6600-final-moba
This is my project for 6600 **Dumoba**

The main result of current stage is under 'dumoba_3v5_mode' and the structure is:

```
├── dumoba_env.py          # The fully custom MOBA environment
├── train_ppo_3v5.py       # PPO 
├── eval.py                # Evaluation + GIF generation
├── models/                # Saved model checkpoints
└── results/               # Auto-generated GIFs and summary reports
```

Under that file, run

```
python train_ppo_3v5.py
```
It's gonna train the ppo with default settings, you can choose the steps taken to train and the difficulty mode. 

After you have the model, if you want to visualize and see the stats, run

```
python eval.py
```

and the results will be stored under

```
results/<map_type>/<difficulty>/summary.txt
results/<map_type>/<difficulty>/*.gif
```

At this moment the maptype function is only avaiable for classic due to time limits. 