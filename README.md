# Arabic Dialects Identification

There are many Arabic countries, and every country has its own dialect, 
some of them are similar and some are totally different.  

In this repo I've created 2 machine learning models (LSTM model, and a pipeline of a TFIDF Victorizer and a Random Forest) to classify 18 dialects.   


To be able to run the app and the notebooks, you should have [`daar`]("https://pypi.org/project/daar/"), a tiny package  I've created especially for this task, you can install it with all other packages using `pip install -r requirements.txt`

To run the app, you have 2 options:
- Option_1:
    - `git clone https://github.com/Ahmed-Elshehawy/arabic-dialects-identification.git`
    - `cd "arabic-dialects-identification"`
    - `pip install -r requirements.txt` 
    - `mkdir app/no_stem/models` # create new directory(models) inside app/no_stem/
    - download [LSTM model]("https://www.kaggleusercontent.com/kf/90088475/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..bNTluRhq_tzXJ1OinjDx_w.905yaFn0eZbWJuUOuP2YmRbJZx6a03LjRmmfyjtUByDS1shO-QDUVvu11EyGqtPgmitfIerY3rEty3bqVb1lton5cpf5LDKD7k3sdeq85QlkHAXRL0BO5nKj2OFb9EKPCDQOIsqnb9l1hDzgzO35tFNVb41Co4yDQPW4-FfTv-xa4UH3Jy02jBKdlG0LBMILL16DTsjuOTYpuEjNyXu3VSEzPtK1hQ4Qzn2E1AJjo54dG3I50fPtlMaN70WdSiMufQiH-LleI-qn64AFJ5ro303GAcz29GKxKe8gdjREnp7FpdMOkhFRQyGxi_Fww40-jyDhvwfM4Fzr1GyMYVqzeaWNi3o7pmO2kQGJIuQm22BFPPFGZ9ZQjWQbsGvmiMtmNUYpXApMlQ5S8ZKZwLq-6DWQkhkS90a7ol9YxG3-v0ukmN32cxIwmQLipEN2bNhoTRNhWAEiht6PHjAEC5O6niL-zjEDBkh0NzgnUlDGsDWjj2miO3aEQPF63pVjdOm8MvSKKieJ-2uocF0HbHlDvrwzU7xNVVY9-fzlGixLqIzfRDCIyVqa6WE3vKrLF4tfzs5VcQkrvX34F1WNiO7IIg_7Mq_-Vkw2t9PEF1KC-4wMw_UlzjJrtPcA5PG0rD9Xco1w57H4y5xlkNK6GwStFg.f2sBlnVkKfy4NpDbGdabfw/best_model.pt")  and [ML model]("https://www.kaggleusercontent.com/kf/89869800/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..OZ2dUYyGTkJF6CCa170AoA.mUXtJ00pAtBWwO2pu5jaU9xe4AU0lyMOdCwCZqMHsjoRhq1k9E5Ez-GInE99OMptGkz9UUTBDpCk4Iu8wQGK_W9pHF3aK9r3BK0Y8W73tQSFcO3pjCWquz1fMjcp-yCqw7HXKqRQmvOkbTAbCXge_UT_FtaXlTUHMIiIAEMqG_Za2M1WW9Bfn4KUUUp-ZwDy_KmEGzq_4ro0HL1KWiwxgcNLGP2jshQ6aQSCAcY-5qmb8hjO5SScY95qYfgpb_SINYDpwNwma59JzPmMe0JdklvYfSZaOoVeDfq1IK9AVbaMakntAL8hpfucd-JtI5yksGmAJZicQuGapbnbas0k4gyNmzomuNO6fCIxV9gtJjyNYFLsmwLtQnQgUoA8B2hBvbssNSZQPoyjzI1FCeFOOtpNDzD72G3ZbgeYlsM3mx19B81lKqrpfIMvsdNxJNwbUPcr5F1KgvK_wXzMJE4af85zs7ibebbi9oLXZOR4gGfcsjJOUtb8lubDlwDD2dmE9KFrs8b0tKiV8veZ5ME5OLhfC_NrjGe-DN0TwO8EHHDii-texDm79kFXPIudhIoahs4O2uXRd3rtTVi3c1x9aEBq9vzAI-YT1buOq8ywpnHIHL7R2YKKsdK75F1y49102HWFXFithybGf6Gu1m4SpfVKEBUYZRpBx_5QUSqcIGI.xebZ48yt-Lw2bwYgJe8_aA/pipe_rf_20.obj") in this directory
    - run `python3 app.py`

- Optin_2: 
    - download `Dockerfile`
    - in the same directory run `docker build -t <your image name> .`
    - run `docker create -p <your port number>:1111 --name <your container name> <your image name>`
    - run `docker start <your container name>` # to start the app
    - open `http://localhost:<your port number>/` in your browser
    - to stop the app: run docker `stop <your container name>`

## After running app (it may take some time to start):
- you can upload any csv file (*must have column called "text"*), you can test it using `sample.csv` file in this repo
- select the model you want
- a plot with dialect cout will appear on the screen, 
- you can download a csv file `result.csv` containing 2 columns 
    - text: the original text
    - pred_dialect: the predicted dialect
- also, you can download the plot by hovering it and click download
- **In addition to that you can test the model by typing your own text**

## Example:

![img](assets/img1.png)
