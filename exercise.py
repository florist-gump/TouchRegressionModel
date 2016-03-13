import numpy as np
import matplotlib.pyplot as plt
import sys

K_FOLDS = 10

class model:
    def __init__(self, x_w, x_order, y_w, y_order):
        self.x_w = x_w
        self.x_order = x_order
        self.y_w = y_w
        self.y_order = y_order

    def predict(self,x,y):
        #normalize y
        y = 1 - y
        x_offset = predict_offset(self.x_order, self.x_w, np.array([[x]]))
        y_offset = predict_offset(self.y_order, self.y_w, np.array([[y]]))
        return x + x_offset[0,0] , 1 - y + y_offset[0,0]

#remove Subject from input so the user_id can be converted
def remove_subject_string_from_input_file(input_file_name, output_file_name):
    with open(input_file_name, 'r') as infile, open(output_file_name, 'w') as outfile:
        data = infile.read()
        data = data.replace("Subject", "")
        outfile.write(data)

def calc_w(order, touch, offset):
    X = np.hstack((np.ones_like(touch), touch))
    for i in range(2, order + 1, 1):
        X = np.c_[X, touch ** i]
    term1 = np.dot(X.transpose(), X)
    term1_inv = np.linalg.inv(term1)
    term2 = np.dot(X.transpose(), offset)
    return np.dot(term1_inv, term2)

def calc_avg_loss(order, training_touch, training_offset, validation_touch, validation_offset, w=None):
    if w is None:
        w = calc_w(order, training_touch, training_offset)

    predicted_offset = predict_offset(order, w, validation_touch)

    loss = abs(validation_offset - predicted_offset)
    loss = np.mean(loss)
    return loss

def predict_offset(order, w, touch):
    X_test = np.hstack((np.ones_like(touch), touch))
    for i in range(2, order + 1, 1):
        X_test = np.c_[X_test, touch ** i]
    predicted_offset = np.dot(X_test, w)
    return predicted_offset


def plot_model_with_data(model, data):
    target = np.column_stack((data[:, 0][:, None], 1 - data[:, 1][:, None]))
    touch = np.column_stack((data[:, 2][:, None], 1 - data[:, 3][:, None]))

    x_target = target[:, 0][:, None]
    x_touch = touch[:, 0][:, None]
    y_target = target[:, 1][:, None]
    y_touch = touch[:, 1][:, None]


    e = x_touch - x_target
    f = y_touch - y_target

    show_touch = x_touch[::150]
    offset_prediction_x = predict_offset(model.x_order, model.x_w, show_touch)

    plt.plot(show_touch, offset_prediction_x)
    plt.plot(x_touch, e, 'ro')

    plt.xlabel('touch')
    plt.ylabel('offset')
    plt.title('target/offset')
    plt.grid(True)
    plt.show()

def illustrate_best_order(orders, losses):
    plt.plot(orders, losses)

    plt.xlabel('Polynominal Order')
    plt.ylabel('Average CV Loss')
    plt.title('Performance of different orders')
    plt.show()

def illustrate_user_specific_model_against_global_model(user_model_losses, global_model_losses, user_ids):
    plt.plot(user_ids, user_model_losses, marker='o', color='r', ls='', label='user specific model')
    plt.plot(user_ids, global_model_losses, marker='o', color='b', ls='', label='global model')

    plt.xlabel('User Ids')
    plt.ylabel('Average CV Loss')
    plt.axis([plt.xlim()[0]-1,plt.xlim()[1]+1, plt.ylim()[0],plt.ylim()[1]])
    plt.show()



def cross_validate(touch_data, offset_data, model_order, folds, w=None):
    fold_size = touch_data.__len__()/folds
    losses = []

    for fold in range(0, folds):
        val_fold_start = fold * fold_size
        val_fold_stop = val_fold_start + fold_size
        validation_indices = np.arange(val_fold_start, val_fold_stop)

        training_touch = np.delete(touch_data, validation_indices)[:, None]
        training_offset = np.delete(offset_data, validation_indices)[:, None]
        validation_touch = np.concatenate((touch_data[:val_fold_start], touch_data[val_fold_stop:]))
        validation_offset = np.concatenate((offset_data[:val_fold_start], offset_data[val_fold_stop:]))

        curr_loss = calc_avg_loss(model_order, training_touch, training_offset, validation_touch, validation_offset, w)
        losses.append(curr_loss)

    return np.mean(losses)


def find_best_order(touch_data, offset_data, folds):
    best_order = 1
    min_loss = sys.maxint
    losses = []
    highest_order_to_test = 3

    #test orders up to X
    for curr_order in range(1, highest_order_to_test, 1):
        loss = cross_validate(touch_data, offset_data, curr_order, folds)
        losses.append(loss)
        if loss < min_loss:
            min_loss = loss
            best_order = curr_order

    #illustrate_best_order(range(1, highest_order_to_test, 1), losses)
    return best_order



def generate_model(data):
    #shuffle input data
    np.random.shuffle(data)
    target = np.column_stack((data[:, 0][:, None], 1 - data[:, 1][:, None]))
    touch = np.column_stack((data[:, 2][:, None], 1 - data[:, 3][:, None]))

    x_target = target[:, 0][:, None]
    x_touch = touch[:, 0][:, None]
    y_target = target[:, 1][:, None]
    y_touch = touch[:, 1][:, None]


    e = x_touch - x_target
    f = y_touch - y_target

    x_order = find_best_order(x_touch, e, K_FOLDS)
    y_order = find_best_order(y_touch, f, K_FOLDS)

    x_w = calc_w(x_order,x_touch,e)
    y_w = calc_w(y_order,y_touch,e)

    return model(x_w,x_order,y_w,y_order)

def test_model(data, model):
    #shuffle input data
    np.random.shuffle(data)
    target = np.column_stack((data[:, 0][:, None], 1 - data[:, 1][:, None]))
    touch = np.column_stack((data[:, 2][:, None], 1 - data[:, 3][:, None]))


    x_target = target[:, 0][:, None]
    x_touch = touch[:, 0][:, None]
    y_target = target[:, 1][:, None]
    y_touch = touch[:, 1][:, None]

    e = x_touch - x_target
    f = y_touch - y_target

    loss_x = cross_validate(x_target, e, model.x_order, K_FOLDS, model.x_w)
    loss_y = cross_validate(y_target, f, model.y_order, K_FOLDS, model.y_w)

    return (loss_x + loss_y)/2

# return compare average losses
def compare_models(model_1, model_2 , data):
    model_1_loss = test_model(data, model_1)
    model_2_loss = test_model(data, model_2)
    return model_1_loss - model_2_loss

#only return data from a certain user
def select_specific_user_data(data,user_id):
    return data[data[:,4] == user_id]


def main():
    user_ids = [15,19,23,24]
    #research question: Are individual specific models better than models trained with data from a collection of users?
    #remove subject string form input so the id can be converted
    remove_subject_string_from_input_file('courseworkdata.csv', 'courseworkdata_cleaned.csv')
    data = np.loadtxt('courseworkdata_cleaned.csv', delimiter=',', skiprows=1, usecols={1, 2, 3, 4, 5})
    #generate a model with data from all users

    global_model = generate_model(data)
    user_models = []

    plot_model_with_data(global_model, data)

    #for user_id in user_ids:
    #    user_models.append(generate_model(select_specific_user_data(data,user_id)))

    #user_model_losses = []
    #global_model_losses = []
    #global_performance = 0
    #for i,user_model in enumerate(user_models):
    #    user_model_losses.append(test_model(select_specific_user_data(data,user_ids[i]),user_model))
    #    global_model_losses.append(test_model(select_specific_user_data(data,user_ids[i]),global_model))
    #    global_performance += compare_models(global_model, user_model, select_specific_user_data(data,user_ids[i]))

    #print global_performance/user_models.__len__()
    #illustrate_user_specific_model_against_global_model(user_model_losses, global_model_losses, user_ids)

    #print g_model.predict(0.034375,0.564606741573034)
    #print test_model(select_specific_user_data(data,user_ids[0]),g_model)-test_model(select_specific_user_data(data,user_ids[0]),u_model)



if __name__ == "__main__":
    main()
