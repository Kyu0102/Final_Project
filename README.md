# Final_Project

제목

움직임 감지기

설명

영상 속 물체의 움직임을 감지한다. 이때 움직임을 감지한 부분을 crop해 trained CNN model를 활용해 classification을 수행한다.
이를 통해 감지한 움직인 물체가 무엇인지를 확인한다.

구현 방법:
1. CNN model 중 classification을 수행할 수 있는 model을 선정한다.
2. 영상을 입력으로 받아 모니터에 display할 수 있게 만든다.
3. 영상에서 움직임을 감지하고 detection 박스를 만들어 crop한다.
4. crop된 이미지를 model에 입력으로 넣어 classification을 수행한다.
5. 움직임이 감지된 물체가 무엇인지 확인할 수 있는 움직임 감지기 완성!

-최종 구현-

![image](https://github.com/Kyu0102/Final_Project/assets/128031528/af67b0d0-09e0-4f75-9044-182830f3e90b)

-위와 같이 움직이는 물체를 크롭한 후 이를 MobileNetV2에 입력으로 넣어 inference 결과를 box위에 글자로 표시한다.

-parameter가 굉장히 적은 MobileNetV2를 사용하고 inference 횟수를 real-time이 아닌 어느정도 딜레이를 줌으로써 CPU 환경에서도 real-time처럼 작동한다.

-원래는 CIFAR100을 학습해 inference를 수행할 예정이었으나, CIFAR100으로 학습한 모델은 classification 성능이 이보다도 떨어져서 이는 보류하기로 했다.

-현재 사용하는 데이터셋은 ImageNet이다.

-Drawback-

1. classification 성능이 별로라 인식이 불안정하다.
2. 움직이는 물체에 대해 box를 만들 때 너무 작은 움직임에 대해서는 box를 만들지 못한다.
3. 움직이는 물체에 대해 1개의 box만 만들어야 하나, mask끼리 연결되지 않아 여러 개의 box를 만드는 경우가 있다.

-Reference

1. https://webnautes.tistory.com/1248
