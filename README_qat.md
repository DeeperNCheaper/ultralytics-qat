# yolov8-qat
purpose: this is for applying QAT to yolov8

- [ ] 환경 구축

- [ ] Dockerfile: pytorch-quantization install, volume mount, 

- [ ] install requirements 

- [ ] trainer

- [ ] validator

- [ ] Exporter

- [ ] trainer.py, validator.py copy & paste해서 사용할 수 있게 만들기.

model.train(args) -> DetectionTrainer(BaseTrainer)

- 데이터 저장소 볼륨 마운트

python3 train.py --data=coco128.yaml --model=yolov8n.pt --epochs=1 --batch=8 --device=0 --qat

