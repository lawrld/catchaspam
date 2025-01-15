clc;

%% load dataset
data = readtable('spam.csv','TextType', 'string');
data.v1 = categorical(data.v1);

%% preprocess text data
cvp = cvpartition(data.v1, 'HoldOut', 0.3);
dataTrain = data(training(cvp), :);
dataValidation = data(test(cvp), :);

%% text preprocessing
textDataTrain = dataTrain.v2;
textDataValidation = dataValidation.v2;
labelsTrain = dataTrain.v1;
labelsValidation = dataValidation.v1;

%% tokenize text data
documentsTrain = tokenizedDocument(textDataTrain);
documentsValidation = tokenizedDocument(textDataValidation);

%% create word encoding
enc = wordEncoding(documentsTrain);

%% convert text data to sequences
XTrain = doc2sequence(enc, documentsTrain);
XValidation = doc2sequence(enc, documentsValidation);

%% define lstm architecture
inputSize = 1;
embeddingDimension = 50;
numHiddenUnits = 90;

layers = [... 
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(embeddingDimension, enc.NumWords)
    lstmLayer(numHiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

%% training options
% train sdgm

optionsSGDM = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.03, ...
    'MaxEpochs', 5, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XValidation, labelsValidation}, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

netSGDM = trainNetwork(XTrain, labelsTrain, layers, optionsSGDM);

% Train Adam
optionsAdam = trainingOptions('adam', ...
    'InitialLearnRate', 0.03, ...
    'MaxEpochs', 5, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XValidation, labelsValidation}, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

netAdam = trainNetwork(XTrain, labelsTrain, netSGDM.Layers, optionsAdam);

% test the model on new messages
testMessages = ["U dun say so early hor... U c already then say...", ...
    "Nah I dont think he goes to usf, he lives around here though", ...
    "FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs send, å£1.50 to rcv", ...
    "Even my brother is not like to speak with me. They treat me like aids patent."];

% tokenize text messages
documentsTest = tokenizedDocument(testMessages);

% convert test data to sequences
testSequences = doc2sequence(enc, documentsTest);

% classify test messages
predictedLabels = classify(netAdam, testSequences);

% display predicted labels
for i = 1:length(testMessages)
    fprintf('Message: %s\nPredicted Label: %s\n\n', testMessages(i), string(predictedLabels(i)));
end
