array = [101, 45, 2, 6, 5, 12, 2, 51, 3, 4, 0, 11, 12, 9, 32, 5, 22, 90]

def bubble_sort(array):
    # Неоптимальный вариант (всегда n^2)
    for i in range(len(array)):
        for i in range(1, len(array)):
            if (array[i-1] > array[i]):
                temp = array[i]
                array[i] = array[i-1]
                array[i-1] = temp
                isSwapped = True

def enhanced_bubble_sort(array):
    # Оптимальный вариант (кол-во итераций зависит от изначального расположения элементов в массиве)
    while True:
        for i in range(1, len(array)):
            if i == 1:
                isSwapped = False
            if array[i-1] > array[i]:
                temp = array[i]
                array[i] = array[i-1]
                array[i-1] = temp
                isSwapped = True
        if not isSwapped:
            break

def selection_sort(array):
    return array
    
def insertion_sort(array):
    return array

def quick_sort(array):
    return array

print('before:', array)
enhanced_bubble_sort(array)
print('after', array)

# print(bubble_sort(array))
# print(selection_sort(array))
# print(insertion_sort(array))
