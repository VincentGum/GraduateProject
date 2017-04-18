from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
import tensorflow as tf

last_momentum0 = []
last_momentum1 = []
last_momentum2 = []
last_momentum3 = []
last_n0 = []
last_n1 = []
last_n2 = []
last_n3 = []


def Ada(loss, parameter_list):

    opt = GradientDescentOptimizer(1e-3)
    grads_and_vars = opt.compute_gradients(loss, parameter_list)
    capped_grads_and_vars = []

    for i in range(len(grads_and_vars)):
        gradient = grads_and_vars[i][0]
        variable = grads_and_vars[i][1]
        if len(last_n0)!=0:
            n = tf.multiply(gradient, gradient) + last_n0[i]
            momentum = gradient/(tf.sqrt(n) + 0.001)
            capped_grads_and_vars.append((momentum, variable))

        else:
            n = tf.multiply(gradient, gradient)
            momentum = gradient / (tf.sqrt(n) + 0.001)
            capped_grads_and_vars.append((momentum, variable))

    if len(last_n0) != 0:
        for i in range(len(grads_and_vars)):
            last_n0[i] = capped_grads_and_vars[i][0]
    else:
        for i in range(len(grads_and_vars)):
            last_n0.append(capped_grads_and_vars[i][0])

    return opt.apply_gradients(grads_and_vars)



def Ada_Mom(loss, parameter_list):

    mu = 0.9  # the parameter of the momentum, always be 0.9
    opt = GradientDescentOptimizer(1e-3)
    grads_and_vars = opt.compute_gradients(loss, parameter_list)
    capped_grads_and_vars = []

    for i in range(len(grads_and_vars)):
        gradient = grads_and_vars[i][0]
        variable = grads_and_vars[i][1]
        if len(last_n1) != 0:
            n = tf.multiply(gradient, gradient) + last_n1[i]
            momentum = 0.9 * last_momentum1[i] + gradient/(tf.sqrt(n) + 0.001)
            capped_grads_and_vars.append((momentum, variable))
        else:
            n = tf.multiply(gradient, gradient)
            momentum = gradient / (tf.sqrt(n) + 0.001)
            capped_grads_and_vars.append((momentum, variable))

    if len(last_momentum1) != 0:
        for i in range(len(grads_and_vars)):
            last_momentum1[i] = capped_grads_and_vars[i][0]
    else:
        for i in range(len(grads_and_vars)):
            last_momentum1.append(capped_grads_and_vars[i][0])

    if len(last_n1) != 0:
        for i in range(len(grads_and_vars)):
            last_n1[i] = capped_grads_and_vars[i][0]
    else:
        for i in range(len(grads_and_vars)):
            last_n1.append(capped_grads_and_vars[i][0])

    return opt.apply_gradients(grads_and_vars)

def RMSProp(loss,parameter_list):

    opt = GradientDescentOptimizer(1e-3)
    grads_and_vars = opt.compute_gradients(loss, parameter_list)
    capped_grads_and_vars = []

    for i in range(len(grads_and_vars)):
        gradient = grads_and_vars[i][0]
        variable = grads_and_vars[i][1]
        if len(last_n2) != 0:
            n = 0.8 * tf.multiply(gradient, gradient) + 0.2 * last_n2[i]
            momentum = gradient/(tf.sqrt(n) + 0.001)
            capped_grads_and_vars.append((momentum, variable))
        else:
            n = tf.multiply(gradient, gradient)
            momentum = gradient / (tf.sqrt(n) + 0.001)
            capped_grads_and_vars.append((momentum, variable))

    if len(last_n2) != 0:
        for i in range(len(grads_and_vars)):
            last_n2[i] = capped_grads_and_vars[i][0]
    else:
        for i in range(len(grads_and_vars)):
            last_n2.append(capped_grads_and_vars[i][0])

    return opt.apply_gradients(grads_and_vars)



def RMSProp_Mom(loss, parameter_list):

    mu = 0.9  # the parameter of the momentum, always be 0.9
    opt = GradientDescentOptimizer(1e-3)
    grads_and_vars = opt.compute_gradients(loss, parameter_list)
    capped_grads_and_vars = []

    for i in range(len(grads_and_vars)):
        gradient = grads_and_vars[i][0]
        variable = grads_and_vars[i][1]
        if len(last_n3) != 0:
            n = 0.8 * tf.multiply(gradient, gradient) + 0.2 * last_n3[i]
            momentum = 0.9 * last_momentum3[i] + gradient/(tf.sqrt(n) + 0.001)
            capped_grads_and_vars.append((momentum, variable))
        else:
            n = tf.multiply(gradient, gradient)
            momentum = gradient / (tf.sqrt(n) + 0.001)
            capped_grads_and_vars.append((momentum, variable))

    if len(last_momentum3) != 0:
        for i in range(len(grads_and_vars)):
            last_momentum3[i] = capped_grads_and_vars[i][0]
    else:
        for i in range(len(grads_and_vars)):
            last_momentum3.append(capped_grads_and_vars[i][0])

    if len(last_n3) != 0:
        for i in range(len(grads_and_vars)):
            last_n3[i] = capped_grads_and_vars[i][0]
    else:
        for i in range(len(grads_and_vars)):
            last_n3.append(capped_grads_and_vars[i][0])

    return opt.apply_gradients(grads_and_vars)
