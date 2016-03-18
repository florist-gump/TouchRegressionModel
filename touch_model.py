import numpy as np
import matplotlib.pyplot as plt
import sys
# set the amount of k-folds used during cross validation
K_FOLDS = 10
# flag to set if a model is used that treats the offset depending on x and y separately or on both together
USE_2D_MODEL = True
PLOT_RESULTS = True
# print results on console
DEBUG_MODE = True
# shuffle the input data set before doing the cross validation
SHUFFLE_INPUT_DATA = False
USER_IDS = [15,19,23,24]

# class to store the offset prediction model, defined by the polynomial order and the w parameter
class model:
    def __init__(self, x_w, x_order, y_w, y_order):
        self.x_w = x_w
        self.x_order = x_order
        self.y_w = y_w
        self.y_order = y_order
    # make predictions for a given touch
    def predict(self,_x,_y):
        #normalize y
        y = 1 - _y
        y = np.array([[y]])
        x = np.array([[_x]])
        if USE_2D_MODEL:
            x_offset = predict_offset(self.x_order, self.x_w, touch_wrapper(x))
            y_offset = predict_offset(self.y_order, self.y_w, touch_wrapper(y))
        else:
            x_offset = predict_offset(self.x_order, self.x_w, touch_wrapper([x,y],True))
            y_offset = predict_offset(self.y_order, self.y_w, touch_wrapper([y,x],True))
        return _x + x_offset[0,0] , 1 - _y + y_offset[0,0]

# class to wrap touch
# can either be contain both x and y (is_2d = true)
# or just contain x or y (is_2d = false)
class touch_wrapper:
    def __init__(self, touch, is_2d=False):
        self.touch = touch
        self.is_2d = is_2d
    # when 2d touch return corresponding item, if 1d just return x or y
    def __getitem__(self, item):
        if self.is_2d:
            return self.touch[item]
        else:
            return self.touch

#remove "Subject" from input so the user_id can be converted
def remove_subject_string_from_input_file(input_file_name, output_file_name):
    with open(input_file_name, 'r') as infile, open(output_file_name, 'w') as outfile:
        data = infile.read()
        data = data.replace("Subject", "")
        outfile.write(data)

# calc X matrix (used in calculating w)
# is different for a 1d and 2d touch
def calc_X(order, touch):
    X = None
    if not touch.is_2d:
        touch = touch[0]
        X = np.hstack((np.ones_like(touch), touch))
        for i in range(2, order + 1, 1):
            X = np.c_[X, touch ** i]
    else:
        X = np.hstack((np.ones_like(touch[0]), touch[0]))
        X = np.c_[X, touch[1]]
        for i in range(2, order + 1, 1):
            X = np.c_[X, touch[0] ** i]
            X = np.c_[X, touch[1] ** i]
            # model the interaction effect
            if i == 2:
                X = np.c_[X, touch[0] * touch[1]]
            else:
                X = np.c_[X, (touch[0] ** (i)) * (touch[1] ** (i-1))]
                X = np.c_[X, (touch[1] ** (i)) * (touch[0] ** (i-1))]
    return X

# calculate w
def calc_w(order, touch, offset):
    X = calc_X(order, touch)
    term1 = np.dot(X.transpose(), X)
    term1_inv = np.linalg.inv(term1)
    term2 = np.dot(X.transpose(), offset)
    return np.dot(term1_inv, term2)

# calculate the average
def calc_avg_loss(order, training_touch, training_offset, validation_touch, validation_offset, w=None):
    if w is None:
        w = calc_w(order, training_touch, training_offset)

    predicted_offset = predict_offset(order, w, validation_touch)

    loss = (validation_offset - predicted_offset)**2
    loss = np.mean(loss)
    return loss

# predict the offset given a polynomial and a touch location
def predict_offset(order, w, touch):
    X_test = calc_X(order, touch)
    predicted_offset = np.dot(X_test, w)
    return predicted_offset

# plot the touch location and the modeled offset
def plot_model_with_data(model, data):
    target = np.column_stack((data[:, 0][:, None], 1 - data[:, 1][:, None]))
    touch = np.column_stack((data[:, 2][:, None], 1 - data[:, 3][:, None]))

    x_target = target[:, 0][:, None]
    x_touch = touch[:, 0][:, None]
    y_target = target[:, 1][:, None]
    y_touch = touch[:, 1][:, None]


    e = x_touch - x_target
    f = y_touch - y_target

    predict_for_x = np.linspace(0, 1.0, 100)[:, None]
    if USE_2D_MODEL:
        offset_prediction_x = predict_offset(model.x_order, model.x_w, touch_wrapper(predict_for_x))
    else:
        predict_for_y = np.linspace(0, 0.5, 100)[:, None]
        offset_prediction_x = predict_offset(model.x_order, model.x_w, touch_wrapper([predict_for_x,predict_for_y], True))

    plt.plot(predict_for_x, offset_prediction_x, lw=2)
    plt.plot(x_touch, e, 'ro', ms=4)

    plt.xlabel('Touch')
    plt.ylabel('Offset')
    plt.title('Touch and Offset for x')
    plt.grid(True)
    plt.show()


    predict_for_y = np.linspace(0, 0.5, 100)[:, None]
    if USE_2D_MODEL:
        offset_prediction_y = predict_offset(model.y_order, model.y_w, touch_wrapper(predict_for_y))
    else:
        predict_for_x = np.linspace(0, 1.0, 100)[:, None]
        offset_prediction_y = predict_offset(model.y_order, model.y_w, touch_wrapper([predict_for_y,predict_for_x], True))

    plt.plot(predict_for_y, offset_prediction_y, lw=2)
    plt.plot(y_touch, f, 'ro', ms=4)

    plt.xlabel('Touch')
    plt.ylabel('Offset')
    plt.title('Touch and Offset for y')
    plt.grid(True)
    plt.show()

# displays the average loss for a list of losses, representing different polynomials
def illustrate_best_order(orders, losses):
    plt.plot(orders, losses)
    plt.xlabel('Polynomial Order')
    plt.ylabel('Average CV Loss')
    plt.title('Average CV loss of polynomial orders')
    plt.show()

def illustrate_user_specific_model_against_global_model(user_model_losses, global_model_losses, user_ids):
    plt.plot(user_ids, user_model_losses, marker='o', color='r', ls='', label='user-specific model')
    plt.plot(user_ids, global_model_losses, marker='o', color='b', ls='', label='non-user-specific model')

    plt.xlabel('User Ids')
    plt.ylabel('Average CV Loss')
    plt.axis([plt.xlim()[0]-1,plt.xlim()[1]+1, plt.ylim()[0],plt.ylim()[1]+plt.ylim()[1]*0.20])
    plt.legend()
    plt.show()

# do cross validation used for training and model testing
def cross_validate(touch_data, offset_data, model_order, folds, w=None):
    fold_size = touch_data[0].__len__()/folds
    losses = []

    for fold in range(0, folds):
        val_fold_start = fold * fold_size
        val_fold_stop = val_fold_start + fold_size
        validation_indices = np.arange(val_fold_start, val_fold_stop+1)

        # generate training and validation data
        if not touch_data.is_2d:
            training_touch = touch_wrapper(np.delete(touch_data[0], validation_indices)[:, None])
        else:
            training_touch = touch_wrapper([np.delete(touch_data[0], validation_indices)[:, None],np.delete(touch_data[1], validation_indices)[:, None]], True)
        training_offset = np.delete(offset_data, validation_indices)[:, None]
        if not touch_data.is_2d:
            validation_touch = np.concatenate((touch_data[0][:val_fold_start], touch_data[0][val_fold_stop:]))
            validation_touch = touch_wrapper(validation_touch)
        else:
            validation_touch_1 = np.concatenate((touch_data[0][:val_fold_start], touch_data[0][val_fold_stop:]))
            validation_touch_2 = np.concatenate((touch_data[1][:val_fold_start], touch_data[1][val_fold_stop:]))
            validation_touch = touch_wrapper([validation_touch_1,validation_touch_2], True)
        validation_offset = np.concatenate((offset_data[:val_fold_start], offset_data[val_fold_stop:]))

        curr_loss = calc_avg_loss(model_order, training_touch, training_offset, validation_touch, validation_offset, w)
        losses.append(curr_loss)

    return np.mean(losses)

# find the best order, where best is defined by having the lowest avg loss
def find_best_order(touch_data, offset_data, folds):
    best_order = 1
    min_loss = sys.maxint
    losses = []
    highest_order_to_test = 13

    #test orders up to K_FOLDS
    for curr_order in range(1, highest_order_to_test, 1):
        avg_loss = cross_validate(touch_data, offset_data, curr_order, folds)
        losses.append(avg_loss)
        if avg_loss < min_loss:
            min_loss = avg_loss
            best_order = curr_order

    #if PLOT_RESULTS:
    #    illustrate_best_order(range(1, highest_order_to_test, 1), losses)
    return best_order


# generates a model based on the given input data
# can be used to generate global or user specific model
def generate_model(data):
    if SHUFFLE_INPUT_DATA:
        np.random.shuffle(data)
    target = np.column_stack((data[:, 0][:, None], 1 - data[:, 1][:, None]))
    touch = np.column_stack((data[:, 2][:, None], 1 - data[:, 3][:, None]))

    x_target = target[:, 0][:, None]
    x_touch = touch[:, 0][:, None]
    y_target = target[:, 1][:, None]
    y_touch = touch[:, 1][:, None]

    e = x_touch - x_target
    f = y_touch - y_target

    if USE_2D_MODEL:
        x_order = find_best_order(touch_wrapper(x_touch), e, K_FOLDS)
        y_order = find_best_order(touch_wrapper(y_touch), f, K_FOLDS)
    else:
        x_order = find_best_order(touch_wrapper([x_touch,y_touch], True), e, K_FOLDS)
        y_order = find_best_order(touch_wrapper([y_touch,x_touch], True), f, K_FOLDS)

    if USE_2D_MODEL:
        x_w = calc_w(x_order,touch_wrapper(x_touch),e)
        y_w = calc_w(y_order,touch_wrapper(y_touch),f)
    else:
        x_w = calc_w(x_order,touch_wrapper([x_touch,y_touch], True),e)
        y_w = calc_w(y_order,touch_wrapper([y_touch,x_touch], True),f)

    return model(x_w,x_order,y_w,y_order)

# tests a model with on the given input touch data
def test_model(data, model):
    if SHUFFLE_INPUT_DATA:
        np.random.shuffle(data)
    target = np.column_stack((data[:, 0][:, None], 1 - data[:, 1][:, None]))
    touch = np.column_stack((data[:, 2][:, None], 1 - data[:, 3][:, None]))


    x_target = target[:, 0][:, None]
    x_touch = touch[:, 0][:, None]
    y_target = target[:, 1][:, None]
    y_touch = touch[:, 1][:, None]

    e = x_touch - x_target
    f = y_touch - y_target

    if USE_2D_MODEL:
        avg_loss_x = cross_validate(touch_wrapper(x_target), e, model.x_order, K_FOLDS, model.x_w)
        avg_loss_y = cross_validate(touch_wrapper(y_target), f, model.y_order, K_FOLDS, model.y_w)
    else:
        avg_loss_x = cross_validate(touch_wrapper([x_target,y_target], True), e, model.x_order, K_FOLDS, model.x_w)
        avg_loss_y = cross_validate(touch_wrapper([y_target,x_target], True), f, model.y_order, K_FOLDS, model.y_w)

    return (avg_loss_x + avg_loss_y)/2

# return compare by returning the difference in average loss over the input touch data
def compare_models(model_1, model_2 , data):
    model_1_loss = test_model(data, model_1)
    model_2_loss = test_model(data, model_2)
    return model_1_loss - model_2_loss

#only return data from a certain user
def select_specific_user_data(data,user_id):
    return data[data[:,4] == user_id]

# print a comparison of losses of two models in percentage on the command line
def print_loss_comparison(loss1, loss2, userids=None):
    loss_percentages = [(100/a * b)-100 for a, b in zip(loss1, loss2)]
    for i,loss in enumerate(loss_percentages):
        if userids is None:
            print loss
        else:
            print '{0}: {1}'.format(userids[i], loss)

def main():
    #remove subject string form input so the user id can be converted
    remove_subject_string_from_input_file('courseworkdata.csv', 'courseworkdata_cleaned.csv')
    data = np.loadtxt('courseworkdata_cleaned.csv', delimiter=',', skiprows=1, usecols={1, 2, 3, 4, 5})

    #generate a global model with data from all users
    global_model = generate_model(data)

    # for illustration purposes
    if DEBUG_MODE:
        print 'average loss of global model: {0}'.format(test_model(data, global_model))
    if PLOT_RESULTS:
        plot_model_with_data(global_model, data)

    # list of user models trained only on data corresponding to each user
    user_models = []
    for user_id in USER_IDS:
        user_models.append(generate_model(select_specific_user_data(data,user_id)))

    # compare user models against global model
    user_model_losses = []
    global_model_losses = []
    for i,user_model in enumerate(user_models):
        user_model_losses.append(test_model(select_specific_user_data(data,USER_IDS[i]),user_model))
        global_model_losses.append(test_model(select_specific_user_data(data,USER_IDS[i]),global_model))

    # illustrate results
    if PLOT_RESULTS:
        illustrate_user_specific_model_against_global_model(user_model_losses, global_model_losses, USER_IDS)
    if DEBUG_MODE:
        print 'improvements of user models compared to global model in %:'
        print_loss_comparison(user_model_losses, global_model_losses, USER_IDS)

    # compare a specific user models against other user models
    # i.e. testing a user model for user 1 with data from user 2,3 and 4
    other_than_x_user_model_losses = []
    user_x_model_losses = []
    user_x = 0
    for i,user_model in enumerate(user_models):
        #leave out testing user model against itself
        if not i == user_x:
            other_than_x_user_model_losses.append(test_model(select_specific_user_data(data,USER_IDS[i]),user_model))
            user_x_model_losses.append(test_model(select_specific_user_data(data,USER_IDS[user_x]),user_models[user_x]))

    # illustrate results
    USER_IDS.pop(user_x)
    if PLOT_RESULTS:
        illustrate_user_specific_model_against_global_model(other_than_x_user_model_losses, user_x_model_losses , USER_IDS)
    if DEBUG_MODE:
        print 'improvements of user a user specific model compared to a model of another user in %:'
        print_loss_comparison(other_than_x_user_model_losses, user_x_model_losses, USER_IDS)

    # example on how to use the model to make a prediction for a given touch
    x = 0.034375
    y = 0.5646067
    x_pred_g,y_pred_g = global_model.predict(x,y)
    x_pred_u,y_pred_u = global_model.predict(x,y)
    if DEBUG_MODE:
        print 'given x: {0}, y: {1}'.format(x,y)
        print 'the predictions are x: {0}, y: {1} in the global model'.format(x_pred_g,y_pred_g)
        print 'the predictions are x: {0}, y: {1} in model for user 1'.format(x_pred_u,y_pred_u)

if __name__ == "__main__":
    main()