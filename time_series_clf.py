from recurent_nn import *

def find_min_t(X_test):
    min_t = 10**6
    for handwritting in X_test:
        if len(handwritting) < min_t:
            min_t = len(handwritting)

    return min_t

def export_results2(score,acc):
    
    ExportToFile="rnn-test-time-series-clf"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv"
    folder = 'D:\\handwritten-analysis\\results-test\\'
    
    prediction = pd.DataFrame(list(zip(score, acc)),
               columns =['scores', 'accuracy'])
    # prediction.index = X_test.index # its important for comparison
    prediction.to_csv(folder+ExportToFile)
    print(prediction)
    return


def real_time(model,X_train, Y_train, X_test, Y_test):
    score_list = []
    accuracy_list = []
    train_index_list = []
    test_index_list = []
    parameters = []
    epochs = 1
    
    print('real_time')
    min_t = find_min_t(X_test)
    step =  min_t // 3

    for i in range(1,min_t,step):
        new_test_X = X_test[:i]
        new_test_Y = Y_test[:i] 

        model.fit(X_train, Y_train, 
                    epochs=epochs, batch_size=1,
                    validation_data=(new_test_X, new_test_Y))

        score,acc = model.evaluate(new_test_X, new_test_Y, verbose = 2)
        print("Logloss score: %.2f" % (score))
        print("Validation set Accuracy: %.2f" % (acc))

        score_list.append(score)
        accuracy_list.append(acc)

        export_results2(score_list, accuracy_list)




    return