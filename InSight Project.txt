#
# https://automatetheboringstuff.com/chapter13/ 
# http://theautomatic.net/2019/10/14/how-to-read-word-documents-with-python/
# https://stackoverflow.com/questions/52125333/extracting-bold-text-from-resumes-docx-doc-pdf-using-python
#
##useful commands
# del
# type()

# Helper packages.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Packages with tools for text processing.
import nltk
import docx
# Packages for working with text data.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# Packages for getting data ready for and building a LDA model
import gensim
from gensim import corpora, models
from pprint import pprint
from gensim.models.coherencemodel import CoherenceModel

# Set Working Directory
main_dir = "C:/Users/TruongQ/DSF-Project"
data_dir = main_dir +  "\\data"
# Change the working directory
os.chdir(data_dir)
# Check the working directory
print(os.getcwd())


###just some experimentation with the python-docx package
#doc = docx.Document("MandE suggestions.docx")
#len(doc.paragraphs)
#doc.paragraphs[0].text
#doc.paragraphs[1].text
#doc.paragraphs[2].text
#doc.paragraphs[3].text
#
#doc.paragraphs[1].runs[0].text
#doc.paragraphs[1].runs[1].text
#doc.paragraphs[1].runs[2].text
#doc.paragraphs[1].runs[3].text

#### import text 
def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

##this is just an alternative method to import text...however, it puts the entire document on a single string
#import readDocx
#yy = readDocx.getText('Abt Final Revised Technical Proposal 4-29-19.docx')
#del yy


#### turn text into a df
doc = docx.Document("SHOPS Plus FP_Year 2 Annual report_FINAL_10.31.19.docx")
doc = docx.Document("HFG HF-RMNCH FY18Q2 Report_For Submission.docx")
doc = docx.Document("HFG Y6 Q3 WEST AFRICA REGIONAL Quarterly Performance Monitoring Report.docx")
doc = docx.Document("Abt Final Revised Technical Proposal 4-29-19.docx")

result = [p.text for p in doc.paragraphs]
df = pd.DataFrame(result, columns = ['text'])
del result

#### add helper columns
#i should add some code to delete the initial junk so that exec summary has a low % through
df['extent'] = np.arange(len(df))/len(df)
df['length'] = df['text'].str.len()

#### make plot/histogram of length
plt.scatter(df.index,
            df['length'], 
            marker = "h") 
plt.show()

plt.bar(df.index, df["length"])
plt.xlabel('index')                
plt.ylabel('length')
plt.title('Plot of text density by location in text') 
plt.show()

#### try to figure out how to make this into a word cloud
from wordcloud import WordCloud
import nltk
import nltk.data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Create a series from the dataframe
text = df["text"]
print(text[:5])

# Tokenize each title into a large list of tokenized titles.
text_tokenized = [word_tokenize(text[i]) for i in range(0,len(text))]

## This is just a test run of the cleaning code.
#text_words = text_tokenized[1]
#print(text_words)
## 1. Convert to lower case.
#text_words = [word.lower() for word in text_words]
#print(text_words[:10])
## 2. Remove stopwords.
## Get common English stop words.
#stop_words = set(stopwords.words('english'))
## Remove stop words.
#text_words = [word for word in text_words if not word in stop_words]
#print(text_words[:10])
## 3. Remove punctuation and any non-alphabetical characters.
#text_words = [word for word in text_words if word.isalpha()]
#print(text_words[:10])
## 4. Stem words.
#text_words = [PorterStemmer().stem(word) for word in text_words]
#print(text_words[:10])


## Create a vector for clean text.
text_clean = [None] * len(text_tokenized)

# Create a vector of word counts for each clean text.
word_counts_per_text = [None] * len(text_tokenized)

# Process words in all documents.
stop_words = set(stopwords.words('english'))

for i in range(len(text_tokenized)):
    # 1. Convert to lower case.
    text_clean[i] = [text.lower() for text in text_tokenized[i]]
    
    # 2. Remove stopwords.
    text_clean[i] = [word for word in text_clean[i] if not word in stop_words]
    
    # 3. Remove punctuation and any non-alphabetical characters.
    text_clean[i] = [word for word in text_clean[i] if word.isalpha()]
    
    # 4. Stem words.
    text_clean[i] = [PorterStemmer().stem(word) for word in text_clean[i]]
    
    # Record the word count per text.
    word_counts_per_text[i] = len(text_clean[i])
    
# Histogram 
plt.hist(word_counts_per_text, bins = len(set(word_counts_per_text)))
plt.xlabel('Number of words per text')
plt.ylabel('Frequency')

# Array with length of each text.
word_counts_array = np.array(word_counts_per_text)
text_array = np.array(text_clean)
print(len(text_array))

# Find indices of all messages where there are at least 3 words.
valid_text = np.where(word_counts_array >= 3)[0]
print(len(valid_text))

# Subset the text array to keep only those where there are at least 5 words.
text_array = text_array[valid_text]
print(len(text_array))

# Convert the array back to a list.
text_clean = text_array.tolist()
print(text_clean[:10])

# Join words in each message into a single character string.
text_clean_list = [' '.join(message) for message in text_clean]
print(text_clean_list[:10])

del word_counts_array, valid_text, text_array 
del word_counts_per_text, text_tokenized, text_clean, text

# create the DTM out of the cleaned text list
vec = CountVectorizer()
X = vec.fit_transform(text_clean_list)
exercise_DTM = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
print(exercise_DTM)

# Convert the matrix into a pandas dataframe for easier manipulation.
exercise_DTM = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
print(exercise_DTM.head())


def HeadDict(dict_x, n):
    # Get items from the dictionary and sort them by
    # value key in descending (i.e. reverse) order.
    sorted_x = sorted(dict_x.items(),
    reverse = True,
    key = lambda kv: kv[1])
    # Convert sorted dictionary to a list.
    dict_x_list = list(sorted_x)
    # Return the first `n` values from the dictionary only.
    return(dict(dict_x_list[:n]))
# Sum the counts of each word in all documents and save the series as a dictionary.
    
# Sum frequencies of each word in all documents.
exercise_DTM.sum(axis = 0).head()

# Save series as a dictonary.
corpus_freq_dist = exercise_DTM.sum(axis = 0).to_dict()

# Glance at the top 30 words with highest counts.
print(HeadDict(corpus_freq_dist, 30))

# Save as a FreqDist object native to nltk.
corpus_freq_dist = nltk.FreqDist(corpus_freq_dist)

# Plot distribution for the entire corpus.
plt.figure(figsize = (16, 7))
corpus_freq_dist.plot(80)

#make the word cloud
wordcloud = WordCloud(max_font_size = 40, background_color = "white")
wordcloud = wordcloud.generate(' '.join(text_clean_list))

# Plot the cloud using matplotlib.
plt.figure(figsize = (14, 7))
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.show()

del exercise_DTM, corpus_freq_dist, text_clean_list

####### create more covariates
#whether it is right after a line break
afterbreak = df["length"]==0
afterbreak = pd.concat([pd.Series([True]), afterbreak])
afterbreak = afterbreak.drop(afterbreak.index[len(afterbreak)-1])
afterbreak = afterbreak.tolist()
df["afterbreak"] = afterbreak
del afterbreak

#searching for terms
df['lower']= df['text'].str.lower()
df['exec_summary']= df['lower'].str.find('executive summary') 

#font name/size.....this gives a lot of false numbers...font should actually be mean of all run fontsizes...
#https://github.com/python-openxml/python-docx/issues/409

#doc.paragraphs[13].style.font.name
#doc.paragraphs[1].style.font.name
#doc.paragraphs[2].style.font.name
#doc.paragraphs[3].style.font.name

fontname=[]
for p in doc.paragraphs:
    name = p.style.font.name
    size = p.style.font.size
    fontname.append(name)

fontsize=[]
for p in doc.paragraphs:
    name = p.style.font.name
    size = p.style.font.size
    fontsize.append(size)

df['fontname'] = fontname
df['fontsize'] = fontsize
del fontname, fontsize

# % of letters that are capitalized


#whether it has a heading 1 or heading 0
### optional: scrape the document for headings...you can do a search/find later on the df to mark the headings
heading1 = []
for paragraph in doc.paragraphs:
    if paragraph.style.name == 'Heading 1':
        heading1.append(paragraph.text)
df['heading1'] = df['text'].isin(heading1)

heading2 = []
for paragraph in doc.paragraphs:
    if paragraph.style.name == 'Heading 2':
        heading2.append(paragraph.text)
df['heading2'] = df['text'].isin(heading2)
del heading1, heading2

#whether it has a bold or italic run
#### optional: scrape the document for bold/italic text. Note: sometimes bold comes from heading rather than bold
#bolds=[]
#italics=[]
#for para in doc.paragraphs:
#    for run in para.runs:
#        if run.italic :
#            italics.append(run.text)
#        if run.bold :
#            bolds.append(run.text)
#boltalic_Dict={'bold_phrases':bolds,
#              'italic_phrases':italics}
#del bolds, italics, boltalic_Dict

####### create y variable
df['y'] = 0
df.loc[df["heading1"], ["y"]] = 1
df.loc[df["heading2"], ["y"]] = 1

##### Try to output certain sections based on breaks

series = pd.Series(df['y']) 
df["cumulative"] = series.cumsum() 

output = (df.groupby('cumulative')['text'].apply("\n".join).reset_index())
output = output.drop(df.index[0])
output = output.reset_index()

qq = df.loc[df["y"]==1, ["text"]]
output['title'] = qq["text"].tolist()

#### produce report

from docx import Document
from docx.shared import Inches

document = Document()

document.add_heading("Quang's Report Generator Example", 0)

p = document.add_paragraph('This is an example of a ')
p.add_run('DSF Tool ').bold = True
p.add_run('that can be used to generate reports. ')
p.add_run("Let's generate an example USAID report using the HFG data that we scraped from the earlier Python code.").italic = True

document.add_paragraph('Here is the LHSS Theory of Change:', style='Intense Quote')
document.add_picture('LHSS ToC.jpg', width=Inches(5.25))

for i in range(len(output["text"])):
    document.add_heading(output["title"][i], level=1)
    document.add_paragraph(output["text"][i], style='List Bullet')
    document.add_paragraph('first item in ordered list', style='List Number')

records = (
    (3, '101', 'Spam'),
    (7, '422', 'Eggs'),
    (4, '631', 'Spam, spam, eggs, and spam')
)

table = document.add_table(rows=1, cols=3)
hdr_cells = table.rows[0].cells
hdr_cells[0].text = 'Qty'
hdr_cells[1].text = 'Id'
hdr_cells[2].text = 'Desc'
for qty, id, desc in records:
    row_cells = table.add_row().cells
    row_cells[0].text = str(qty)
    row_cells[1].text = id
    row_cells[2].text = desc

#document.add_page_break()
document.save('demo.docx')


#### run a logistic regression on
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Split into x and y. y is categorical, so it can't scale.
ex_X = df[['extent','length', 'afterbreak', 'fontsize','heading1','heading2']] #removed fontname for causing issues
ex_y = np.array(df['y'])

#fixing fontsize since the nan's will cause issues
#imputing the mean, but this is how you would impute the mode: https://stackoverflow.com/questions/42789324/pandas-fillna-mode/42789818
print(ex_X.isnull().sum())
ex_X= ex_X.fillna(ex_X.mean())

# Set the seed & split into train and test.
np.random.seed(0)

ex_X_train, ex_X_test, ex_y_train, ex_y_test = train_test_split(ex_X, 
                                                    ex_y, 
                                                    test_size=0.3) 

# Set up logistic regression model.
ex_logistic_regression_model = linear_model.LogisticRegression()
print(ex_logistic_regression_model)

# Fit the model.
ex_logistic_regression_model.fit(ex_X_train, 
                                 ex_y_train)

# Predict on test data.
ex_predicted_values = ex_logistic_regression_model.predict(ex_X_test)
print(ex_predicted_values)

#create confusion matrix
ex_conf_matrix = metrics.confusion_matrix(ex_y_test, ex_predicted_values)
print(ex_conf_matrix)

#calculate accuracy
ex_test_accuracy = metrics.accuracy_score(ex_y_test, ex_predicted_values)
print("Accuracy on test data: ", ex_test_accuracy)

#see model scores
ex_model_scores = pd.DataFrame(columns=['metrics', 'values', 'model'])

ex_model_scores = ex_model_scores.append({'metrics' : "accuracy" , 
                                  'values' : round(ex_test_accuracy,4),
                                  'model':'logistic' } , 
                                  ignore_index=True)
print(ex_model_scores)

##ROC
# Get probabilities instead of predicted values.
ex_test_probabilities = ex_logistic_regression_model.predict_proba(ex_X_test)
print(ex_test_probabilities[0:5, :])

# Get probabilities of test predictions only.
test_predictions = ex_test_probabilities[:, 1]
print(test_predictions[0:5])

# Get FPR, TPR, and threshold values.
fpr, tpr, threshold = metrics.roc_curve(ex_y_test,         #<- test data labels
                                        test_predictions)  #<- predicted probabilities

# Get AUC by providing the FPR and TPR.
auc = metrics.auc(fpr, tpr)
print("Area under the ROC curve: ", auc)

# Make an ROC curve plot.
plt.title('Receiver Operator Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


##get accuracy report
# Create a list of target names to interpret class assignments.
target_names = ['Not a Section', 'Section']

# Print an entire classification report.
class_report = metrics.classification_report(ex_y_test, 
                                             ex_predicted_values, 
                                             target_names = target_names)
print(class_report)

#### improve the model by using penalty weighting
from sklearn.model_selection import GridSearchCV
dummy_X = df[['extent','length', 'afterbreak', 'fontsize','heading1','heading2']] #removed fontname for causing issues
dummy_y = np.array(df['y'])

print(dummy_X.isnull().sum())
dummy_X= dummy_X.fillna(dummy_X.mean())


np.random.seed(0)
dummy_X_train, dummy_X_test, dummy_y_train, dummy_y_test = train_test_split(dummy_X, 
                                                    dummy_y, 
                                                    test_size = 0.3) 
 
# Create regularization penalty space.
penalty = ['l1', 'l2']
# Create regularization constant space.
C = np.logspace(0, 10, 10)
print("Regularization constant: ", C)
# Create hyperparameter options dictionary.
hyperparameters = dict(C = C, penalty = penalty)
print(hyperparameters)
# Grid search 20-fold cross-validation with above parameters.
clf = GridSearchCV(linear_model.LogisticRegression(), #<- function to optimize
                   hyperparameters,                   #<- grid search parameters
                   cv = 20,                           #<- 10-fold cv
                   verbose = 0)          
# Fit CV grid search.
ex_best_model = clf.fit(dummy_X_train, dummy_y_train)
ex_best_model
# Get best penalty and constant parameters.
penalty = ex_best_model.best_estimator_.get_params()['penalty']
constant = ex_best_model.best_estimator_.get_params()['C']
print('Best penalty: ', penalty)
print('Best C: ', constant)

# Predict on test data using best model.
best_predicted_values = ex_best_model.predict(dummy_X_test)
print(best_predicted_values)
# Compute best model accuracy score.
best_accuracy_score = metrics.accuracy_score(dummy_y_test, best_predicted_values)
print("Accuracy on test data (best model): ", best_accuracy_score)
# Get probabilities instead of predicted values.
best_test_probabilities = ex_best_model.predict_proba(dummy_X_test)
print(best_test_probabilities[0:5, ])
# Get probabilities of test predictions only.
best_test_predictions = best_test_probabilities[:, 1]
print(best_test_predictions[0:5])
# Get ROC curve metrics.
best_fpr, best_tpr, best_threshold = metrics.roc_curve(dummy_y_test, best_test_predictions)
best_auc = metrics.auc(best_fpr, best_tpr)
print(best_auc)
# Make an ROC curve plot.
plt.title('Receiver Operator Characteristic')
plt.plot(fpr, tpr, 'blue', 
         label = 'AUC = %0.2f'%auc)
plt.plot(best_fpr, best_tpr, 'black', 
         label = 'AUC (best) = %0.2f'%best_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

#################### test the same model on another document

doc = docx.Document("HFG HF-RMNCH FY18Q2 Report_For Submission.docx")

result = [p.text for p in doc.paragraphs]
df = pd.DataFrame(result, columns = ['text'])
del result

#### add helper columns
df['extent'] = np.arange(len(df))/len(df)
df['length'] = df['text'].str.len()

####### create more covariates
#whether it is right after a line break
afterbreak = df["length"]==0
afterbreak = pd.concat([pd.Series([True]), afterbreak])
afterbreak = afterbreak.drop(afterbreak.index[len(afterbreak)-1])
afterbreak = afterbreak.tolist()
df["afterbreak"] = afterbreak
del afterbreak

#searching for terms
df['lower']= df['text'].str.lower()
df['exec_summary']= df['lower'].str.find('executive summary') 

#font name/size.....this gives a lot of false numbers...font should actually be mean of all run fontsizes...
#https://github.com/python-openxml/python-docx/issues/409

#doc.paragraphs[13].style.font.name
#doc.paragraphs[1].style.font.name
#doc.paragraphs[2].style.font.name
#doc.paragraphs[3].style.font.name

fontname=[]
for p in doc.paragraphs:
    name = p.style.font.name
    size = p.style.font.size
    fontname.append(name)

fontsize=[]
for p in doc.paragraphs:
    name = p.style.font.name
    size = p.style.font.size
    fontsize.append(size)

df['fontname'] = fontname
df['fontsize'] = fontsize
del fontname, fontsize

# % of letters that are capitalized


#whether it has a heading 1 or heading 0
### optional: scrape the document for headings...you can do a search/find later on the df to mark the headings
heading1 = []
for paragraph in doc.paragraphs:
    if paragraph.style.name == 'Heading 1':
        heading1.append(paragraph.text)
df['heading1'] = df['text'].isin(heading1)

heading2 = []
for paragraph in doc.paragraphs:
    if paragraph.style.name == 'Heading 2':
        heading2.append(paragraph.text)
df['heading2'] = df['text'].isin(heading2)
del heading1, heading2

### create y
df['y'] = 0
df['y'][1] = 1
df['y'][13] = 1
df['y'][105] = 1
df['y'][121] = 1
df['y'][125] = 1
df['y'][130] = 1

dummy_y_test = np.array(df['y'])

dummy_X_test = df[['extent','length', 'afterbreak', 'fontsize','heading1','heading2']] #removed fontname for causing issues
print(dummy_X_test.isnull().sum())
dummy_X_test= dummy_X_test.fillna(dummy_X_test.mean())


# Predict on test data using best model.
best_predicted_values = ex_best_model.predict(dummy_X_test)
print(best_predicted_values)
# Compute best model accuracy score.
best_accuracy_score = metrics.accuracy_score(dummy_y_test, best_predicted_values)
print("Accuracy on test data (best model): ", best_accuracy_score)
# Get probabilities instead of predicted values.
best_test_probabilities = ex_best_model.predict_proba(dummy_X_test)
print(best_test_probabilities[0:5, ])
# Get probabilities of test predictions only.
best_test_predictions = best_test_probabilities[:, 1]
print(best_test_predictions[0:5])
# Get ROC curve metrics.
best_fpr, best_tpr, best_threshold = metrics.roc_curve(dummy_y_test, best_test_predictions)
best_auc = metrics.auc(best_fpr, best_tpr)
print(best_auc)
# Make an ROC curve plot.
plt.title('Receiver Operator Characteristic')
plt.plot(best_fpr, best_tpr, 'black', 
         label = 'AUC (best) = %0.2f'%best_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

#create confusion matrix
ex_conf_matrix = metrics.confusion_matrix(dummy_y_test, best_predicted_values)
print(ex_conf_matrix)


### uses 
# outputting to multiple report formats (even combining multiple reports into one)
# searching within a document for skillsets and then using a match algorithm to match that against Abt employees
# searching Abt's history/archives to find patterns (ie are we balanced in the sectors we shoot for?)...do a sentiment analysis


