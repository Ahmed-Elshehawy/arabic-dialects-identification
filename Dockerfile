FROM ubuntu
WORKDIR /usr/app



RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y python3.8 python3-distutils python3-pip python3-apt

RUN apt-get install git -y
RUN apt install wget -y

RUN git clone "https://github.com/Ahmed-Elshehawy/arabic-dialects-identification.git"


RUN pip install -r "arabic-dialects-identification/requirements.txt"


RUN mkdir "arabic-dialects-identification/app/no_stem/models"

RUN wget "https://www.kaggleusercontent.com/kf/90088475/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..bNTluRhq_tzXJ1OinjDx_w.905yaFn0eZbWJuUOuP2YmRbJZx6a03LjRmmfyjtUByDS1shO-QDUVvu11EyGqtPgmitfIerY3rEty3bqVb1lton5cpf5LDKD7k3sdeq85QlkHAXRL0BO5nKj2OFb9EKPCDQOIsqnb9l1hDzgzO35tFNVb41Co4yDQPW4-FfTv-xa4UH3Jy02jBKdlG0LBMILL16DTsjuOTYpuEjNyXu3VSEzPtK1hQ4Qzn2E1AJjo54dG3I50fPtlMaN70WdSiMufQiH-LleI-qn64AFJ5ro303GAcz29GKxKe8gdjREnp7FpdMOkhFRQyGxi_Fww40-jyDhvwfM4Fzr1GyMYVqzeaWNi3o7pmO2kQGJIuQm22BFPPFGZ9ZQjWQbsGvmiMtmNUYpXApMlQ5S8ZKZwLq-6DWQkhkS90a7ol9YxG3-v0ukmN32cxIwmQLipEN2bNhoTRNhWAEiht6PHjAEC5O6niL-zjEDBkh0NzgnUlDGsDWjj2miO3aEQPF63pVjdOm8MvSKKieJ-2uocF0HbHlDvrwzU7xNVVY9-fzlGixLqIzfRDCIyVqa6WE3vKrLF4tfzs5VcQkrvX34F1WNiO7IIg_7Mq_-Vkw2t9PEF1KC-4wMw_UlzjJrtPcA5PG0rD9Xco1w57H4y5xlkNK6GwStFg.f2sBlnVkKfy4NpDbGdabfw/best_model.pt" -P "arabic-dialects-identification/app/no_stem/models"



RUN wget "https://www.kaggleusercontent.com/kf/89869800/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..OZ2dUYyGTkJF6CCa170AoA.mUXtJ00pAtBWwO2pu5jaU9xe4AU0lyMOdCwCZqMHsjoRhq1k9E5Ez-GInE99OMptGkz9UUTBDpCk4Iu8wQGK_W9pHF3aK9r3BK0Y8W73tQSFcO3pjCWquz1fMjcp-yCqw7HXKqRQmvOkbTAbCXge_UT_FtaXlTUHMIiIAEMqG_Za2M1WW9Bfn4KUUUp-ZwDy_KmEGzq_4ro0HL1KWiwxgcNLGP2jshQ6aQSCAcY-5qmb8hjO5SScY95qYfgpb_SINYDpwNwma59JzPmMe0JdklvYfSZaOoVeDfq1IK9AVbaMakntAL8hpfucd-JtI5yksGmAJZicQuGapbnbas0k4gyNmzomuNO6fCIxV9gtJjyNYFLsmwLtQnQgUoA8B2hBvbssNSZQPoyjzI1FCeFOOtpNDzD72G3ZbgeYlsM3mx19B81lKqrpfIMvsdNxJNwbUPcr5F1KgvK_wXzMJE4af85zs7ibebbi9oLXZOR4gGfcsjJOUtb8lubDlwDD2dmE9KFrs8b0tKiV8veZ5ME5OLhfC_NrjGe-DN0TwO8EHHDii-texDm79kFXPIudhIoahs4O2uXRd3rtTVi3c1x9aEBq9vzAI-YT1buOq8ywpnHIHL7R2YKKsdK75F1y49102HWFXFithybGf6Gu1m4SpfVKEBUYZRpBx_5QUSqcIGI.xebZ48yt-Lw2bwYgJe8_aA/pipe_rf_20.obj" -P "arabic-dialects-identification/app/no_stem/models"

RUN mv "arabic-dialects-identification/app/no_stem/models/best_model.pt" "arabic-dialects-identification/app/no_stem/models/model.pt"

ENTRYPOINT ["python3"]
CMD ["arabic-dialects-identification/app/app_docker.py"]
