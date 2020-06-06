# Sentiment-Analysis
To directly calculate polarity of a text, run GUI.py-
1. Enter text in the first input field, and click on  calculate button.
2. Lexical Polarity Score and labels from SVM, NB and NN clasifier will be displayed.
3. Enter User data into input fields
4. Click on insert to insert into sql database dtab.db

To run the training scripts-
1. imdb-nn.ipyb: It contains the training code for nueral network.
2. imdb-nb.ipyb: It contains the training code for Naive Bayes classifier.
3. imdb-svm.ipynb: It contains the training code for Support Vector Machine Classifier.

These scripts generate models and also the saved weights produced during training.

The scripts svm_predict, pred_twitter_kaggle and nb_predict load the saved model from files generated after training.

The scripts nb_realtime, svm_realtime and nn_realtime have functions which use trained weigths to provide prediction for
new data.
