import random

array = [101, 45, 2, 6, 5, 12, 2, 51, 3, 4, 0, 11, 12, 9, 32, 5, 22, 90]

def bubble_sort(array):
    # Неоптимальный вариант (всегда n^2)
    for i in range(len(array)):
        for i in range(1, len(array)):
            if (array[i-1] > array[i]):
                array[i], array[i-1] = array[i-1], array[i]

def enhanced_bubble_sort(array):
    # Оптимальный вариант (кол-во итераций зависит от изначального расположения элементов в массиве)
    while True:
        for i in range(1, len(array)):
            if i == 1:
                isSwapped = False
            if array[i-1] > array[i]:
                array[i], array[i-1] = array[i-1], array[i]
                isSwapped = True
        if not isSwapped:
            break

def shake_sort(array):
    start = 0
    end = len(array)
    while start < end:
        isSwapped = False

        for i in range(start + 1, end):
            if array[i-1] > array[i]:
                array[i], array[i-1] = array[i-1], array[i]
                isSwapped = True
        end -= 1

        for i in range(end - 1, start - 1, -1):
            if array[i+1] < array[i]:
                array[i], array[i+1] = array[i+1], array[i]
                isSwapped = True
        start += 1
        
        if not isSwapped: break

def selection_sort(array):
    if len(array) == 0:
        return

    for i in range(len(array)):
        minIdx = i
        min = array[minIdx]
        for j in range(i + 1, len(array)):
            if array[j] < min:
                minIdx = j
                min = array[minIdx]
        array[minIdx], array[i] = array[i], array[minIdx]
                
    
def insertion_sort(array):
    if len(array) == 0:
        return
    
    for i in range(1, len(array)):
        current = array[i]
        j = i - 1

        while j >= 0 and current < array[j]:
            array[j + 1] = array[j]
            j -= 1

        array[j + 1] = current

def quick_sort(array):
    if (len(array) <= 1): return array

    random_index = random.randint(0, len(array) - 1)
    basis = array.pop(random_index)

    greaters = list(filter(lambda x: x > basis, array))
    lowers = list(filter(lambda x: x < basis, array))

    return quick_sort(lowers) + [basis] + quick_sort(greaters)
    

print('before:', array)
shake_sort(array)
print('after', array)
