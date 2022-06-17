for i in 0 1 2 3; do
    nohup python3 yolov5/train.py --epochs 10 --data yolov5/staplers.yaml --weights yolov5n.pt --entity peerteam --project results/DOTA --cache --evolve --device $i > logs/evolve_gpu_$i.log &
done