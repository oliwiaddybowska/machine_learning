import random
import numpy as np

def performInsertionSort(lst):
    compCnt = 0
    for itr1 in range(1, len(lst)): # Loop through the list starting from the second element
        currIndex = itr1 # Set the current index to the current loop iteration
        for itr2 in range(itr1-1, -1, -1): # Loop backwards from the element before itr1 down to the first element
            if lst[currIndex] < lst[itr2]:
                lst[currIndex], lst[itr2] = lst[itr2], lst[currIndex]
                currIndex = itr2
                compCnt += 1

            else: # If no swap is needed
                compCnt += 1
                break

    return lst, compCnt

def performSelectionSort(lst):
    compCnt = 0
    for itr1 in range(0, len(lst) - 1): # Loop through the list from the first element to the second-last element
        minIndex = itr1 # Assume the current index is the minimum
        for itr2 in range(itr1 + 1, len(lst)): # Loop through elements after itr1 to find the actual minimum
            compCnt += 1
            if lst[minIndex] > lst[itr2]:
                minIndex = itr2

        lst[itr1], lst[minIndex] = lst[minIndex], lst[itr1] # Swap the two elements

    return lst, compCnt

def performQuickSort(lst):
    compCnt = 0
    if len(lst) <= 1: # Don't need to compare if list is only 1 element
        return lst, compCnt

    pivotValue = lst[0] # select the first element of list as pivot
    less = []
    greater = []

    for itr in range(1, len(lst)):
        if lst[itr] <= pivotValue:
            less.append(lst[itr])
            compCnt += 1

        elif lst[itr] > pivotValue:
            greater.append(lst[itr])
            compCnt += 1

    lessSort, lessCnt = performQuickSort(less)
    greaterSort, greaterCnt = performQuickSort(greater)

    compCnt += lessCnt
    compCnt += greaterCnt
    result = lessSort + [pivotValue] + greaterSort

    return result, compCnt

N = 10
IstNumbers = list(range(N))
random.shuffle(IstNumbers)
print("List before sorting:", IstNumbers)

IstNumbers = performQuickSort(IstNumbers)
print("List after sorting:", IstNumbers)

def performMergeSort(lst): # Decomposition phase
    compCnt = 0
    if len(lst) == 1:
        return lst, compCnt

    subLstToSort1 = []
    subLstToSort2 = []
    for itr in range(len(lst)):
        if itr < len(lst)/2:
            subLstToSort1.append(lst[itr])
        else:
            subLstToSort2.append(lst[itr])

    lst1, compCnt1 = performMergeSort(subLstToSort1) # Recursion programming
    lst2, compCnt2 = performMergeSort(subLstToSort2)

    compCnt += compCnt1
    compCnt += compCnt2

    result = [] # Aggregation phase
    idx1, idx2 = 0, 0 # Position of elements in each sublist
    for itr in range(len(lst)):
        if idx1 == len(lst1):
            result.append(lst2[idx2])
            idx2 += 1
        elif idx2 == len(lst2):
            result.append(lst1[idx1])
            idx1 += 1
        elif lst1[idx1] < lst2[idx2]:
            result.append(lst1[idx1])
            idx1 += 1
            compCnt += 1
        else:
            result.append(lst2[idx2])
            idx2 += 1
            compCnt += 1

    return result, compCnt


N = 2000
replication = 20

# insertion sorting
lstCompCnt_insert = []
for i in range(replication):
    lstNumbers = list(range(N))
    random.shuffle(lstNumbers)
    result, compCnt = performInsertionSort(lstNumbers)
    lstCompCnt_insert.append(compCnt)

print("!!!! Algorithm analysis for insertion sorting")
print("Average of insertion sorting comparison count:", np.mean(lstCompCnt_insert))
print("Standard deviation of insertion sorting comparison count:", np.std(lstCompCnt_insert))
print(" ")


# selection sorting
lstCompCnt_selection = []
for i in range(replication):
    lstNumbers = list(range(N))
    random.shuffle(lstNumbers)
    result, compCnt = performSelectionSort(lstNumbers)
    lstCompCnt_selection.append(compCnt)

print("!!!! Algorithm analysis for selection sorting")
print("Average of selection sorting comparison count:", np.mean(lstCompCnt_selection))
print("Standard deviation of selection sorting comparison count:", np.std(lstCompCnt_selection))
print(" ")


# quick sorting
lstCompCnt_quick = []
for i in range(replication):
    lstNumbers = list(range(N))
    random.shuffle(lstNumbers)
    result, compCnt = performQuickSort(lstNumbers)
    lstCompCnt_quick.append(compCnt)

print("!!!! Algorithm analysis for quick sorting")
print("Average of quick sorting comparison count:", np.mean(lstCompCnt_quick))
print("Standard deviation of quick sorting comparison count:", np.std(lstCompCnt_quick))
print(" ")


# merge sorting
lstCompCnt_merge = []
for i in range(replication):
    lstNumbers = list(range(N))
    random.shuffle(lstNumbers)
    result, compCnt = performMergeSort(lstNumbers)
    lstCompCnt_merge.append(compCnt)

print("!!!! Algorithm analysis for merge sorting")
print("Average of merge sorting comparison count:", np.mean(lstCompCnt_merge))
print("Standard deviation of merge sorting comparison count:", np.std(lstCompCnt_merge))