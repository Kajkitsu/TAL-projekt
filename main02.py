# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import time
import tracemalloc as tm
from queue import Queue
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gen_dataset_v2(size: int):
    max_value = size * 10
    dataset = np.random.randint(0, max_value, size)
    tsum = sum(np.random.randint(0, max_value, int(size * 0.9)))
    return dataset, tsum, max_value


def accurate_algorithm(dataset: np.ndarray, sum: int):
    dataset = dataset.copy()

    def check(dataset: np.ndarray, sum: int, n: int):
        if sum == 0:
            return True
        elif n == 0 and sum != 0:
            return False
        elif dataset[n - 1] > sum:
            return check(dataset, sum, n - 1)
        return check(dataset, sum, n - 1) or check(dataset, sum - dataset[n - 1], n - 1)

    return check(dataset, sum, dataset.size)


def dynamic_algorithm(dataset: np.ndarray, t_sum: int, if_print=False):
    dataset = dataset.copy()
    dataset.sort()

    def print_subset(t_dataset: np.ndarray, t_subset: np.ndarray, p_sum):
        print(" ")
        print("\t\tsubsum")
        print('number|| ', end="")
        for t_sub_sum in range(0, p_sum + 1):
            print("{0:4d} |".format(t_sub_sum), end="")
        print()
        print('    x || ', end="")
        for t_sub_sum in range(0, p_sum + 1):
            if t_subset[0, t_sub_sum]:
                print("   T |", end="")
            else:
                print("   F |", end="")
        print()

        for t_number_id in range(0, len(t_dataset)):
            print(' {0:4d} || '.format(dataset[t_number_id]), end="")
            for t_sub_sum in range(0, p_sum + 1):
                if t_subset[t_number_id + 1, t_sub_sum]:
                    print("   T |", end="")
                else:
                    print("   F |", end="")
            print()

    subset = np.full((dataset.size + 1, t_sum + 1), False, dtype=bool)
    subset[:, 0] = True
    set_size = len(dataset)

    for number_id in range(1, len(dataset) + 1):
        if dataset[number_id - 1] <= t_sum:
            for subsum in range(1, t_sum + 1):
                if subsum < dataset[number_id - 1]:
                    subset[number_id, subsum] = subset[number_id - 1, subsum]
                else:
                    subset[number_id, subsum] = subset[number_id - 1, subsum] or \
                                                subset[number_id - 1, subsum - dataset[number_id - 1]]
                if subsum == t_sum and subset[number_id, subsum]:
                    return True

    if if_print:
        print_subset(dataset, subset, t_sum)

    return subset[set_size, t_sum]


def if_correct(dataset, t_sum):
    return dynamic_algorithm(dataset, t_sum, False) == accurate_algorithm(dataset, t_sum)


def get_stat_with_threads_and_batch(n: int, sizes, batch):
    results = []
    file_name = str(np.random.randint(10000, 99999))+"tmp_data.csv"
    for x in range(0, int(n / batch)):
        print("Batch: " + str(x) + " started")
        results.append(get_stat_with_threads(batch, sizes))
        pd.concat(results, ignore_index=True).to_csv(file_name)
    if n % batch > 0:
        print("Batch: " + str(int(n / batch)) + " started")
        results.append(get_stat_with_threads(n % batch, sizes))
    return pd.concat(results, ignore_index=True)


def get_stat_with_threads(n: int, sizes):
    que = Queue()
    threads_list = list()

    for x in range(n):
        t = Thread(target=lambda q, arg1: q.put(get_stats(arg1)),
                   args=(que, sizes.copy()))
        t.start()
        print("process: " + str(x) + " started")
        threads_list.append(t)

    # Join all the threads
    counter = 0
    for t in threads_list:
        t.join()
        print("process: " + str(counter) + " finished")
        counter = counter + 1
    # Check thread's return value
    results = []
    while not que.empty():
        results.append(que.get())
    return pd.concat(results, ignore_index=True)


def get_stats(sizes):
    data_time = []
    data_mem = []
    data_correct = []
    data_size = []
    data_result = []
    data_max_value = []
    data_sum = []
    data_algorithm = []
    for size in sizes:
        dataset, t_sum, max_value = gen_dataset_v2(size)
        for algorithm in ["accurate", "dynamic"]:
            t_time, mem, result, correct = get_time_mem_correct_result(dataset, t_sum, algorithm)
            if t_time > 1.0:
                if result:
                    print("\ttim:{0:10.2f} | n:{1:3d} | sum:{2:7d} | max:{3:6d} | res: True  | all: {4} "
                          .format(t_time, size, t_sum, max_value, algorithm))
                else:
                    print("\ttim:{0:10.2f} | n:{1:3d} | sum:{2:7d} | max:{3:6d} | res: False | all: {4} "
                          .format(t_time, size, t_sum, max_value, algorithm))
            # print("t:{0:4f} |".format(time))
            data_result.append(result)
            data_time.append(t_time)
            data_mem.append(mem)
            data_correct.append(correct)
            data_size.append(size)
            data_max_value.append(max_value)
            data_sum.append(t_sum)
            data_algorithm.append(algorithm)
    return pd.DataFrame({
        "time": data_time,
        "mem": data_mem,
        "correct": data_correct,
        "size": data_size,
        "max_value": data_max_value,
        "sum": data_sum,
        "algorithm": data_algorithm,
        "result": data_result
    })


def get_time_mem_correct_result(dataset, t_sum, t_algorithm):
    tm.start()
    start_time = time.time()
    if t_algorithm == "accurate":
        result = accurate_algorithm(dataset, t_sum)
    else:
        result = dynamic_algorithm(dataset, t_sum, False)
    end_time = time.time()
    mem_use = tm.get_traced_memory()[1]
    tm.stop()
    t_time = end_time - start_time
    correct = True if t_algorithm == "accurate" else if_correct(dataset, t_sum)
    return t_time, mem_use, result, correct


def gen_plot(x_label, y_label, dataframe):
    def get_smooth(x, y):
        from scipy.interpolate import make_interp_spline, BSpline
        x_new = np.linspace(x.min(), x.max(), 300)
        spl = make_interp_spline(x, y, k=3)  # type: BSpline
        y_new = spl(x_new)
        return x_new, y_new

    plot_data_acc = dataframe[[x_label, y_label]].where(dataframe["algorithm"] == "accurate").groupby(x_label).mean()
    plot_data_dyn = dataframe[[x_label, y_label]].where(dataframe["algorithm"] == "dynamic").groupby(x_label).mean()
    rol_plot_data_acc = dataframe[[x_label, y_label]].where(dataframe["algorithm"] == "accurate").groupby(
        x_label).mean() \
        .rolling(window=10,
                 center=True).mean()
    rol_plot_data_dyn = dataframe[[x_label, y_label]].where(dataframe["algorithm"] == "dynamic").groupby(x_label).mean() \
        .rolling(window=10,
                 center=True).mean()

    # x_new, y_new = get_smooth(plot_data_acc.index, plot_data_acc[y_label])
    # plt.plot(
    #     x_new,
    #     y_new,
    #     "r-",
    #     label="Accuracy smooth"
    # )
    plt.plot(
        rol_plot_data_acc.index,
        rol_plot_data_acc[y_label],
        "r-",
        label="Accuracy rolling"
    )
    plt.plot(
        plot_data_acc.index,
        plot_data_acc[y_label],
        "r*",
        label="Accuracy"
    )

    # x_new, y_new = get_smooth(plot_data_dyn.index, plot_data_dyn[y_label])
    # plt.plot(
    #     x_new,
    #     y_new,
    #     "b-",
    #     label="Dynamic smooth"
    # )
    plt.plot(
        rol_plot_data_dyn.index,
        rol_plot_data_dyn[y_label],
        "b-",
        label="Dynamic rolling"
    )
    plt.plot(
        plot_data_dyn.index,
        plot_data_dyn[y_label],
        "b*",
        label="Dynamic"
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.grid(True, color='0.95')
    plt.legend(title='Parameter where:')
    plt.title('Plot ' + x_label + ' / ' + y_label)
    plt.savefig('Plot_' + x_label + '-' + y_label + ".png")
    plt.show()


def main():
    # Use a breakpoint in the code line below to debug your script.
    # Press Ctrl+F8 to toggle the breakpoint.
    # %%

    df = get_stat_with_threads_and_batch(
        n=500,
        sizes=[x for x in range(6,27,2)],
        batch=25
    )

    file_name = str(np.random.randint(10000, 99999))+"data.csv"
    df.to_csv(file_name)
    print(df)

    gen_plot("size", "correct", df)
    gen_plot("size", "time", df)
    gen_plot("size", "mem", df)
    gen_plot("max_value", "time", df)
    gen_plot("max_value", "mem", df)
    gen_plot("sum", "time", df)
    gen_plot("sum", "mem", df)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
