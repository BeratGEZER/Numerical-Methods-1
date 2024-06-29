import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.optimize import fsolve
from sympy import diff, symbols

x = np.linspace(0, 10, 1000)

def f_1(x):
    return (x**2 - 11*x + 24.96)

y1 = f_1(x)

plt.figure(figsize=(8,6), facecolor="pink")
plt.plot(x, y1, label="$f_1$", color="red", marker="s", linestyle="solid", linewidth=2, markersize=2)
plt.xlabel("x")
plt.ylabel("f_1(x)")
plt.title("Function f_1(x)")
plt.legend()  
plt.grid()
plt.show()

# When I look at the graph, I see that the first root is closer to 4 in the range (2,4) and the second root is closer to 8 in the range (6,8). I estimate the first root as 3.2 and the second root as 7.3.

# estimation of f_1(x) function roots:
estimation_root=[7.5,3.2]

def find_root(a=1, b=-11, c=24.96):
    delta = b**2 - 4*a*c 
    list_root = []

    if delta > 0:
        root1 = (-b + delta**0.5) / (2*a)
        root2 = (-b - delta**0.5) / (2*a)
        list_root.append(root1)
        list_root.append(root2)
        return list_root
    elif delta == 0:
        root = -b / (2*a)
        list_root.append(root)
        return list_root


table = [["Root Numbers(x)", "Estimated Roots", "Real Roots"]]
root_number = [1, 2]
real_roots = find_root()                                   
est_root = [estimation_root[0], estimation_root[1]]  

# Compare the real roots with the roots estimated by looking at the graph.
for i in range(len(root_number)):
    table.append([root_number[i], est_root[i], real_roots[i]])

print(tabulate(table, headers="firstrow", tablefmt="rst", numalign="center"))

# Bisection function
def bisection(function, number_range):
    tolerance = 0.000000001
    tf = True
    iteration_list = number_range.copy()  
    n = 0
    board_iteration_list = []
    iteration_list_step = []
    while tf:
        n += 1   # iteration steps
        x_1 = iteration_list[0]
        x_2 = iteration_list[1]
        x_3 = (x_1 + x_2) / 2
        board_iteration_list.append(x_3)
        if (function(x_1) > 0 and function(x_3) < 0) or (function(x_1) < 0 and function(x_3) > 0):
            iteration_list.clear()
            iteration_list.append(min(x_1, x_3))
            iteration_list.append(max(x_1, x_3))
            if abs((max(iteration_list) - min(iteration_list)) / 2 ** n) < tolerance:
                break
        elif (function(x_2) > 0 and function(x_3) < 0) or (function(x_2) < 0 and function(x_3) > 0):
            iteration_list.clear()
            iteration_list.append(min(x_2, x_3))
            iteration_list.append(max(x_2, x_3))
            if abs((max(iteration_list) - min(iteration_list)) / 2 ** n) < tolerance:
                break
        else:
            attention = " does not approach the tolerance value."
            return f"{attention}"

    for i in range(n):
        iteration_list_step.append(i) 
           
    table = [["Bisection Iteration Step", "x", "Tolerance"]]
    
    for i in range(len(iteration_list_step)):
        table.append([iteration_list_step[i]+1, board_iteration_list[i], tolerance])
    
    return tabulate(table, headers="firstrow", tablefmt="rst", numalign="center")

print(bisection(f_1, number_range=[6,8]))




def f1_derivative(func, x, h=0.0000001):
    return (func(x + h) - func(x - h)) / (2 * h)   # find any function's derivative


def newton(function, number_range):
    tolerance = 0.000000001
    tf = True
    iteration_list = []
    iteration_list_step = []

    n = 0
    xn = min(number_range)

    while tf:
        n += 1
        x_iteration = xn - function(xn) / f1_derivative(function, xn)
        iteration_list.append(x_iteration)

        if abs(x_iteration - xn) < tolerance:
            break
        else:
            xn = x_iteration

    for i in range(n):
        iteration_list_step.append(i)

    table = [["Newton Iteration Step", "x", "Tolerance"]]

    for i in range(n):
        table.append([i+1, iteration_list[i], tolerance])

    return tabulate(table, headers="firstrow", tablefmt="rst", numalign="center")


print(newton(f_1, number_range=[6,8]))

def secant(function, number_range):
    tolerance = 0.000000001
    tf = True
    iteration_list = []
    iteration_list_step = []

    n = 0
    [xn_1, xn] = number_range

    while tf:
        n += 1
        x_iteration = xn - (function(xn) * (xn - xn_1)) / (function(xn) - function(xn_1))
        iteration_list.append(x_iteration)

        if abs(x_iteration - xn) < tolerance:
            break
        else:
            xn_1, xn = xn, x_iteration

    for i in range(n):
        iteration_list_step.append(i)

    table = [["Secant Iteration Step", "x", "Tolerance"]]

    for i in range(n):
        table.append([i+1, iteration_list[i], tolerance])

    return tabulate(table, headers="firstrow", tablefmt="rst", numalign="center")

print(secant(f_1, [6, 8]))  



def fixed_point_iteration(initial_guess, max_iterations=1000):
    tf = True
    iteration_list = []
    tolerance = 0.000000001
    n = 0
    x = initial_guess

    def f_1(x):
        return (11 * x - 24.96) / x
    

    while tf and n < max_iterations:
        n += 1
        x_iteration = f_1(x)
        iteration_list.append(x_iteration)

        if abs(x_iteration - x) < tolerance:
            break
        else:
            x = x_iteration

    table = [["Fixed Point Iteration Step", "x", "Tolerance"]]

    for i in range(n):
        table.append([i + 1, iteration_list[i], tolerance])

    return tabulate(table, headers="firstrow", tablefmt="rst", numalign="center")


print(fixed_point_iteration(3.5))










def all_methods_and_relative_error(bisection_value, newton_value, secant_value, fixed_pointed_iteration_value, real_root):

    value_list = [bisection_value, newton_value, secant_value, fixed_pointed_iteration_value]
    relative_error_list = []

    for value in value_list:
        relative_error = abs(value - real_root) / real_root  # find relative errors
        relative_error_list.append(relative_error)  # add relative errors in list

    table = [["Method","Your Roots","Real Roots","Relative Error"],["Bisection", value_list[0], real_root, relative_error_list[0]],["Newton", value_list[1], real_root, relative_error_list[1]],["Secant", value_list[2], real_root, relative_error_list[2]],["Fixed Point Iteration", value_list[3], real_root, relative_error_list[3]]]

    return tabulate(table, headers="firstrow", tablefmt="rst", numalign="center")

print(all_methods_and_relative_error(bisection_value=7.80002, newton_value=7.8, secant_value=7.8, fixed_pointed_iteration_value=7.8, real_root=7.8))










def f_2(t):
    return 0.2 * t**2 * (np.cos(2.5*t) * np.exp(-3.2*t) + np.sin(t**2))         # Function for question 2



t = np.linspace(0, 10, 1000)

y2 = f_2(t)



plt.figure(figsize=(8,6), facecolor="yellow")
plt.plot(t, y2, label="$f_2$", color="blue", marker="s", linestyle="solid", linewidth=2, markersize=0.75)
plt.xlabel("t")
plt.ylabel("f_2(t)")
plt.title("Function f_2(t)")
plt.legend()  
plt.grid()
plt.show()


# estimation of f_2(x) function roots:
estimation_root=[1.7,2.5,3.1,3.6,4.0]


def find__roots_2():
    roots = []
    initial_guesses = np.linspace(0, 4, 11)  # Adjust the range of initial guesses
    for guess in initial_guesses:
        root = fsolve(f_2, guess)
        if 0 <= root <= 4:                
            roots.extend(root.tolist())
    roots.sort()  #  sort my finding roots in list 
    return roots


print(find__roots_2())



# When I look at the graph, I detect 5 values ​​that make the f_ 2(x) function zero. my guess is that the roots consist of [1.7, 2.5 3.1 , 3.6 , 4.0 ]









def bisection_2(function, number_range):
    tolerance = 0.000000001
    tf = True
    iteration_list = number_range.copy()
    n = 0
    board_iteration_list = []
    iteration_list_step = []

    while tf:
        n += 1                             # iteration steps
        x_1 = iteration_list[0]
        x_2 = iteration_list[1]
        x_3 = (x_1 + x_2) / 2
        board_iteration_list.append(x_3)
        if (function(x_1) > 0 and function(x_3) < 0) or (function(x_1) < 0 and function(x_3) > 0):
            iteration_list.clear()
            iteration_list.append(min(x_1, x_3))
            iteration_list.append(max(x_1, x_3))
            if abs((max(iteration_list) - min(iteration_list)) / 2 ** n) < tolerance:
                break
        elif (function(x_2) > 0 and function(x_3) < 0) or (function(x_2) < 0 and function(x_3) > 0):
            iteration_list.clear()
            iteration_list.append(min(x_2, x_3))
            iteration_list.append(max(x_2, x_3))
            if abs((max(iteration_list) - min(iteration_list)) / 2 ** n) < tolerance:
                break
        else:
            attention = " does not approach the tolerance value."
            return f"{attention}"

    for i in range(n):
        iteration_list_step.append(i)

    table = [["Bisection Iteration Step", "x", "Tolerance"]]

    for i in range(len(iteration_list_step)):
        table.append([iteration_list_step[i] + 1, board_iteration_list[i], tolerance])

    return tabulate(table, headers="firstrow", tablefmt="rst", numalign="center")




print(bisection_2(f_2, [3.5,3.7]))
print(bisection_2(f_2, [1.5,2.0]))
print(bisection_2(f_2, [3.8,4]))




def f2_derivative(function, t):
    h = 0.0000000000001
    return (function(t + h) - function(t - h)) / (2 * h)

def newton_2(function, number_range):
    tolerance = 0.000000001
    tf = True
    iteration_list = []
    iteration_list_step = []

    n = 0
    xn = min(number_range)

    while tf:
        n += 1
        x_iteration = xn - function(xn) / f2_derivative(function, xn)
        iteration_list.append(x_iteration)

        if abs(x_iteration - xn) < tolerance:
            break
        else:
            xn = x_iteration

    for i in range(n):
        iteration_list_step.append(i)

    table = [["Newton Iteration Step", "x", "Tolerance"]]

    for i in range(n):
        table.append([i+1, iteration_list[i], tolerance])

    return tabulate(table, headers="firstrow", tablefmt="rst", numalign="center")


print(newton_2(f_2, [3.5,3.7]))
print(newton_2(f_2, [1.9,2.6]))
print(newton_2(f_2, [3.9,4.0]))


def secant_2(function, number_range):
    tolerance = 0.000000001
    tf = True
    iteration_list = []
    iteration_list_step = []

    n = 0
    [xn_1, xn] = number_range

    while tf:
        n += 1
        x_iteration = xn - (function(xn) * (xn - xn_1)) / (function(xn) - function(xn_1))
        iteration_list.append(x_iteration)

        if abs(x_iteration - xn) < tolerance:
            break
        else:
            xn_1, xn = xn, x_iteration

    for i in range(n):
        iteration_list_step.append(i)

    table = [["Secant Iteration Step", "x", "Tolerance"]]

    for i in range(n):
        table.append([i+1, iteration_list[i], tolerance])

    return tabulate(table, headers="firstrow", tablefmt="rst", numalign="center")

print(secant_2(f_2, [3.5,3.7]))
print(secant_2(f_2, [1.6,1.8]))
print(secant_2(f_2, [3.9,4.0]))







def fixed_point_iteration(initial_guess):
    tf = True
    iteration_list = []
    tolerance = 0.000000001
    n = 0
    x = initial_guess

    def inverse_f_2(t):
        return np.real(np.sqrt(t / (0.2 * np.cos(2.5*t) * np.exp(-3.2*t) + 0.2 * np.sin(t**2))))

   
        
   
    if abs((inverse_f_2(x) - inverse_f_2(inverse_f_2(x))) / (x - inverse_f_2(x))) < 1 and abs((inverse_f_2(x) - inverse_f_2(inverse_f_2(x))) / (x - inverse_f_2(x))) < 1:    
        while tf:
            n += 1
            x_next = inverse_f_2(x)
            iteration_list.append(x_next)

            if abs(x_next - x) < tolerance:
                break
            else:
                x = x_next

        table = [["Fixed Point Iteration Step", "x", "Tolerance"]]

        for i in range(n):
            table.append([i + 1, iteration_list[i], tolerance])

        return tabulate(table, headers="firstrow", tablefmt="rst", numalign="center")
    else:
        error="invalid value"
        return 

print(fixed_point_iteration(3.5))






def all_methods_and_relative_error_2(bisection_value, newton_value, secant_value, fixed_point_iteration_value, real_root):
    value_list = [bisection_value, newton_value, secant_value, fixed_point_iteration_value]
    relative_error_list = []

    for value in value_list:
        relative_error = abs(value - real_root) / real_root  # find relative errors
        relative_error_list.append(relative_error)  # add relative errors in list

    table = [["Method", "Value", "Real Roots", "Relative Error"],["Bisection", bisection_value, real_root, relative_error_list[0]],["Newton", newton_value, real_root, relative_error_list[1]],["Secant", secant_value, real_root, relative_error_list[2]],["Fixed Point Iteration", fixed_point_iteration_value, real_root, relative_error_list[3]]]

    return tabulate(table, headers="firstrow", tablefmt="rst", numalign="center")


print(all_methods_and_relative_error_2(bisection_value=3.96332, newton_value=3.96333, secant_value=3.96333, fixed_point_iteration_value=3.96333, real_root=3.963326950849822))
print(all_methods_and_relative_error_2(bisection_value=1.77217, newton_value=1.77218, secant_value=1.77218, fixed_point_iteration_value=0, real_root=1.7721835031133988))
