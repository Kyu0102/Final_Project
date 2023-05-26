# Final_Project

-제목
움직임 감지기
-설명
웹캠으로 움직임을 감지한다. 이때 움직임을 감지한 부분을 crop해 trained CNN model를 활용해 classification을 수행한다.
이를 통해 웹캠에서 감지한 움직인 물체가 무엇인지를 확인한다.
구현 방법:
1. CNN model 중 classification을 수행할 수 있는 model을 선정한다.
2. 웹캠을 입력으로 받아 모니터에 display할 수 있게 만든다.
3. 영상에서 움직임을 감지하고 detection 박스를 만들어 crop한다.
4. crop된 이미지를 model에 입력으로 넣어 classification을 수행한다.
5. 움직임이 감지된 물체가 무엇인지 확인할 수 있는 움직임 감지기 완성!
