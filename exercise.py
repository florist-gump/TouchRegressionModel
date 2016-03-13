import numpy as np
import matplotlib.pyplot as plt
import sys

#remove Subject from input so the user_id can be converted
def remove_subject_string_from_input_file(input_file_name, output_file_name):
    with open(input_file_name, 'r') as infile, open(output_file_name, 'w') as outfile:
        data = infile.read()
        data = data.replace("Subject", "")
        outfile.write(data)

def calc_w(order, touch, target):
    X = np.hstack((np.ones_like(touch), touch))
    for i in range(2, order + 1, 1):
        X = np.c_[X, touch ** i]
    term1 = np.dot(X.transpose(), X)
    term1_inv = np.linalg.inv(term1)
    term2 = np.dot(X.transpose(), target)
    return np.dot(term1_inv, term2)

def calc_loss_and_w(order, training_touch, training_target, validation_touch, validation_target):
    w = calc_w(order, training_touch, training_target)
    X_test = np.hstack((np.ones_like(validation_touch), validation_touch))
    for i in range(2, order + 1, 1):
        X_test = np.c_[X_test, validation_touch ** i]
    predicted_offset = np.dot(X_test, w)

    loss = abs(validation_target - predicted_offset)
    loss = np.mean(loss)
    return loss, w


def plot_data(plot_w, plot_k, k_fold_step, touch_data, offset_data, max_plot):
    # show optimal w points
    start_skip = plot_k * k_fold_step
    stop_skip = (plot_k * k_fold_step) + k_fold_step

    training_touch = np.concatenate((touch_data[:start_skip], touch_data[stop_skip:]))
    training_offset = np.concatenate((offset_data[:start_skip], offset_data[stop_skip:]))
    w = calc_w(plot_w, training_touch, training_offset)

    # predict points
    number_points = 100
    x_test = np.linspace(0, max_plot, number_points)[:, None]
    X_TEST = np.hstack((np.ones_like(x_test), x_test))
    for iter_w in range(2, plot_w + 1, 1):
        X_TEST = np.c_[X_TEST, x_test ** iter_w]
    target_predict_x = np.dot(X_TEST, w)

    plt.plot(x_test, target_predict_x)
    plt.plot(training_touch, training_offset, 'ro')

    plt.xlabel('touch')
    plt.ylabel('offset')
    plt.title('target/offset')
    plt.grid(True)
    plt.show()


def cross_validation(touch_data, offset_data, max_plot, folds):
    best_order = 1
    min_loss = sys.maxint
    fold_size = touch_data.__len__()/folds
    best_w = np.array

    for fold in range(0, folds):
        val_fold_start = fold * fold_size
        val_fold_stop = val_fold_start + fold_size
        validation_indices = np.arange(val_fold_start, val_fold_stop)

        training_touch = np.delete(touch_data, validation_indices)[:, None]
        training_offset = np.delete(offset_data, validation_indices)[:, None]
        validation_touch = np.concatenate((touch_data[:val_fold_start], touch_data[val_fold_stop:]))
        validation_offset = np.concatenate((offset_data[:val_fold_start], offset_data[val_fold_stop:]))

        #test orders up to 30
        for curr_order in range(1, 30, 1):
            loss,w = calc_loss_and_w(curr_order, training_touch, training_offset, validation_touch, validation_offset)
            if loss < min_loss:
                min_loss = loss
                best_order = curr_order
                best_w = w

    print "Best: ", best_order, min_loss
    print best_w
    #plot_data(best_order, best_k, k_fold_step, training_touch, training_offset, max_plot)


def generate_model(data):
    #shuffle input data
    np.random.shuffle(data)
    target = np.column_stack((data[:, 0][:, None], 1 - data[:, 1][:, None]))
    touch = np.column_stack((data[:, 2][:, None], 1 - data[:, 3][:, None]))

    y_target = target[:, 1][:, None]
    y_touch = touch[:, 1][:, None]
    x_target = target[:, 0][:, None]
    x_touch = touch[:, 0][:, None]

    e = x_touch - x_target;
    f = y_touch - y_target;

    cross_validation(x_touch, e, 1, 10)
    cross_validation(y_touch, f, 0.5, 10)



def main():
    #research question: Are individual specific models better than models trained with data from a collection of users?
    #remove subject string form input so the id can be converted
    remove_subject_string_from_input_file('courseworkdata.csv', 'courseworkdata_cleaned.csv')
    data = np.loadtxt('courseworkdata_cleaned.csv', delimiter=',', skiprows=1, usecols={1, 2, 3, 4, 5})
    #generate a model with data from all users
    generate_model(data)


if __name__ == "__main__":
    main()
