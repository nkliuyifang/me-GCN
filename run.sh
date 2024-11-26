#nohup python main.py --config config/nturgbd120-cross-subject/joint.yaml --device 0 &
#nohup python main.py --config config/nturgbd120-cross-set/joint.yaml --device 0 &
#nohup python main.py --config config/nturgbd-cross-subject/joint.yaml --device 0 &
#nohup python main.py --config config/nturgbd-cross-view/joint.yaml --device 0 &


python main.py --config config/baseline_ctrgcn_interaction_biBetaInit0.yaml --device 0