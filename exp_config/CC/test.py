
lst = [1,2,3,4,5,6,7,8]
# sum the numbers in the list
print(sum(lst))

# pop the last element of hte list
print(lst.pop())

# copy the list
lst_copy = lst.copy()

# add 1 to each element in the list
lst_copy = [x+1 for x in lst_copy]

# convert every element in the list to string
lst_copy = [str(x) for x in lst_copy]

