import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
from tkinter import simpledialog
from collections import defaultdict
import pickle
import math

import random
import numpy as np
from PIL import Image


timeslotLength = 30 #minutes

numberOfDays = 3

nightBeginsHour = 22 #hours
dayBeginsHour = 8

timeslotsInDay = (24*60)//timeslotLength
planningHorizon = numberOfDays * timeslotsInDay

availableID = 0

dayBeginsMinutes = dayBeginsHour * 60
dayLength = (24-dayBeginsHour)*60

baseLine = [0]*planningHorizon


temp_root = tk.Tk()
temp_root.withdraw()
numberOfDays = simpledialog.askinteger(
    "Настройка расписания", 
    "Введите количество дней (1-7):",
    minvalue=1,
    maxvalue=7
)
temp_root.destroy()

# Обновляем глобальные переменные
timeslotsInDay = (24 * 60) // timeslotLength
planningHorizon = numberOfDays * timeslotsInDay
baseLine = [0] * planningHorizon
availableID = 0  # Сбрасываем ID


class timeString(list):
    def __init__(self, iniList = None):
        if iniList is None:
            iniList = [1] * planningHorizon
            
        if not isinstance(iniList, list):
            raise TypeError(f"can't initialize timeString with {type(iniList).__name__}")
        if len(iniList) != planningHorizon:
            raise ValueError(f"Invalid timeString length: expected {timeslotsInDay}, got {len(iniList)}")
        super().__init__(iniList)
        self.randomize()
    def randomize(self, cost = None):
        if cost is None:
            cost = random.randint(1, planningHorizon)
        elif cost > planningHorizon:
            raise ValueError(f"{cost} is too many for one timeString. must be less then {timeslotsInDay}")
        self[:] = [1] * cost + [0] * (planningHorizon - cost)
        random.shuffle(self)
    def cost(self):
        return sum(self)
    def __sub__(self, other):
        if not isinstance(other, timeString):
            raise TypeError(f"can't subtract {type(other).__name__}, only timeString!")
        res = [ (me or you) - me for me, you in zip(self, other)]
        return timeString(res)

    
def tilZero(string, index):
    count=0
    if string[index]==0:
        return False
    for i in range(index, len(string)):
        if string[i]!=0:
            count+=1
        else:
            break
    return count

def cleanUpNew(string, index, duration):
    i = 0
    while i < len(string):
        if string[i] != 0:  # Found a non-zero sequence
            q = tilZero(string, i)  # Get the length of the sequence
            if q < duration:  # If it's too short, replace it with zeros
                for j in range(i, i + q):
                    string[j] = 0
            i += q  # Skip over the processed sequence
        else:
            i += 1  # Move to the next index if it's already zero
    return string

def vineRight(string, index, duration):
    if string[index] == 0:
        return False
    if duration == 1:
        return True
    if duration <= 0:
        raise ValueError(f"event can't have negative duration, recieved {duration} in right wing")
    canFit = False
    duration=duration-1
    print("i:i+d:d", index, index+duration, duration)
    if index+duration >= len(string):
        return False
    for i in range (0, duration):
        if string[index+duration-i]:
            print(f"index duration i {index+duration-i} {index} {duration} {i}")
            print(f"__canfit tru")
            canFit = True
        else:
            print(f"__canfit FALSE breaking.")
            canFit = False
            break
    return canFit

def vineLeft(string, index, duration):
    index+=1
    if duration == 1:
        return True
    if duration <= 0:
        raise ValueError(f"event can't have negative duration, recieved {duration} in left wing")
    canFit = False
    duration=duration-1
    if index-duration < 0:
        return False
    for i in range(0, duration, 1):
        if string[index-duration+i]:
            canFit = True
        else:
            canFit = False
            break
    return canFit

def vine(string, index, duration):
    if string[index]:
        return vineLeft(string, index, duration) or vineRight(string, index, duration)
    else:
        return False

def canFit(string, index, duration):
    if duration <= 0:
        raise ValueError(f"event can't have negative duration, recieved {duration} in canFit")
    if string[index]:
        if duration == 1:
            return True
        if index+duration > len(string):
            return False
        start_day = index//timeslotsInDay
        end_day = (index + duration) // timeslotsInDay
        if start_day != end_day:
            return False
        result = True
        for i in range(1, duration):
            result = bool( result and string[index+i])
    else: 
        return False
    return result

        

def validityCheck(index, unit, hasID = True):
    beginning=0
    if hasID:
        index+=1
        beginning+=1
    if index < beginning or index >= len(unit):
        if hasID:
            raise ValueError(f"{index-1} is out of bounds of 0..{len(unit)-1}")
        else:
            raise ValueError(f"{index} is out of bounds of 0..{len(unit)}")
    return True
    
class TimeMatrix(np.ndarray):
    def __new__(cls, input_array):
        global availableID
        input_array = np.asarray(input_array)
        
        if input_array.size == 0:
            # Determine shape: empty with 1 column for ID if shape ambiguous
            input_array = input_array.reshape(0, 0)
            obj = np.asarray(input_array).view(cls)
            return obj

        # Ensure 2D shape
        if input_array.ndim == 1:
            input_array = input_array.reshape(-1, 1)

        num_rows = input_array.shape[0]
        id_column = np.arange(availableID, availableID + num_rows).reshape(-1, 1)
        

        availableID += num_rows

        input_array = np.hstack((id_column, input_array))
        
        obj = np.asarray(input_array).view(cls)
        return obj
    
    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, TimeMatrix):  # If slicing gives a TimeMatrix, convert it to np.ndarray
            return result.view(np.ndarray)
        return result
    
    def __mul__(self, other):
        result = super().__mul__(other)  # Perform normal multiplication
        return result
    

    def add_row(self, new_row):
        """Adds a new row to the TimeMatrix with an auto-incremented ID."""
        global availableID

        if len(new_row) != self.shape[1] - 1:  # Ensure new row matches data columns
            raise ValueError(f"New row must have {self.shape[1] - 1} elements.")

        new_id = np.array([[availableID]])  # Create a 2D ID column
        availableID += 1
        new_row = np.array([new_row])  # Convert to a 2D row vector
        new_row = np.hstack((new_id, new_row))  # Add ID column

        updated_matrix = np.vstack((self, new_row)).view(TimeMatrix)
        return updated_matrix  # Return a new TimeMatrix instance
    
    def pop_row(self, index=-1):
        """Remove and return a single row from the matrix by index (default is last row)."""
        if len(self) == 0:
            raise IndexError("Cannot pop from an empty matrix.")
        
        row_to_remove = self[index]
        print("removing")
        self = np.delete(self, index, axis=0).view(TimeMatrix)
        return self, row_to_remove
    
    def kill_row(self, index=-1):
        """Removes a row by index and returns a new TimeMatrix without it."""
        if len(self) == 0:
            raise IndexError("Cannot remove from an empty matrix.")

        return np.delete(self, index, axis=0).view(TimeMatrix)
            
    def surface(self, index, j=0, leave_untouched = False):
        """Return a new array with the specified row on top and the rest below.
        if j is given, surface to this position."""
        if index > len(self):
            raise ValueError(f"Can't surface {index}, can't exceed {len(self)}")
        if index < 0:
            raise ValueError(f"Can't surface {index} because it's negative")
        
        new_matrix = np.vstack([self[index], np.delete(self, index, axis=0)]).view(TimeMatrix)

        row_to_move = self[index]
        
        new_matrix = np.delete(self, index, axis=0)
        
        new_matrix = np.insert(new_matrix, j, row_to_move, axis=0)
        
        new_matrix = new_matrix.view(TimeMatrix)

        if not leave_untouched:
            self[:] = new_matrix
            return self
        
        return new_matrix

    def nullify(self, index, *args):
        # if index < 0 or index >= len(self):
        #     raise ValueError(f"provided row index is out of bounds of {len(self)}")
        validityCheck(index, self, False)
        if len(args)==1:
            actual_column = args[0]+1
            if actual_column > len(self[0]) or actual_column<1:
                raise ValueError(f"can't nullify below {args[0]} since is out of bounds of 1, {len(self[0])}")
            self[index+1:,args[0] +1 ]=0
            return self
        if len(args)==2:
            if args[0]>args[1]:
                raise ValueError(f"Can't nullify from {args[0]} to {args[1]}")
            actual_left_index = args[0]+1
            actual_right_index = args[1]+1
            if actual_left_index<1 or actual_right_index > len(self[0]):
                raise ValueError(f"provided range {args[0], args[1]} is out of bounds of {len(self[0])}")
            self [index+1:, actual_left_index: actual_right_index ]=0
            return self
        else:
            raise ValueError(f"in nullify too few or too many arguments: {args}, expected 1 or 2.")

    def nullifyRow(self, j, *args):
        """j is the index of the row. args either an index or a pair; 
        if index is given than that element nulled; 
        if two indexes then the range nullified including the latter index"""
        validityCheck(j, self, False)
        if len(args)==1:
            self[j][args[0]+1]=0
            return self
        if len(args)==2:
            if args[0]>args[1]:
                raise ValueError(f"Can't nullifyRow from {args[0]} to {args[1]}")
            # if args[0] > len(self[0]) or args[1] > len(self[0]):
            #     raise ValueError(f"index {args[0]} or {args[1]} out of range 0..{len(self[0])}")
            validityCheck(args[1], self[0])
            self[j][args[0]+1:args[1]+1 +1]=0
            return self
        else:
            raise ValueError(f"in nullifyRow too few or too many arguments: {args}, expected 1 or 2.")
    
    def nullifyExcept(self, j, *args):
        """ j — index or the row; args index or a pair """
        validityCheck(j, self, False)
 
        if len(args) == 1:
            user_index = args[0] + 1  # Shift user index by 1
            for col in range(1, len(self[j])):  # Start from 1 to skip ID
                if col != user_index:
                    self[j][col] = 0

        elif len(args) == 2:
            user_start = args[0] + 1  # Shift start index
            user_end = args[1] + 1  # Shift end index
            for col in range(1, len(self[j])):  # Start from 1 to skip ID
                if col < user_start or col >= user_end:
                    self[j][col] = 0
    
    def sumCol(self, i):
        return self[:,i+1].sum().item()
    def sumRow(self, i):
        return self[i,1:].sum().item()
    def countCol(self, i):
        return np.count_nonzero(self[:, i+1])
    def countRow(self,i):
        return np.count_nonzero(self[i, 1:])
    
    
    def cleanUp(self, row, duration):
        i = 1  # Start from index 1 instead of 0
        if duration == 1:
            return self
        while i < len(self[row]):
            if self[row][i] != 0:  # Found a non-zero sequence
                q = tilZero(self[row], i)  # Get the length of the sequence
                if q < duration:  # If it's too short, replace it with zeros
                    for j in range(i, i + q):
                        self[row][j] = 0
                i += q  # Skip over the processed sequence
            else:
                i += 1  # Move to the next index if it's already zero
        return self

    def sortByCost(self):
        for i in range(len(self)):
            min_counter = self.countRow(i)
            min_counter_index = i
            for j in range(i, len(self)):
                current_ = self.countRow(j)
                if current_ < min_counter:
                    min_counter = current_
                    min_counter_index = j
            self.surface(min_counter_index, i)
        return self
    
    def sortByCostBelowRow(self, row):
        for i in range(row + 1, len(self)):  # Начинаем сортировку с row + 1
            min_counter = self.countRow(i)
            min_counter_index = i
            for j in range(i, len(self)):  # Проходим по строкам ниже row
                current_ = self.countRow(j)
                if current_ < min_counter:
                    min_counter = current_
                    min_counter_index = j
            self.surface(min_counter_index, i)  # Обмениваем строки
        return self
    
    def nearestNonZero(self, index, j=1):
        for i in range(j, len(self[0])):
            if self[index][i] != 0:
                return i-1
        return None
    
    def longestStreak(self, row_index):
        max_count = 0
        current_count = 0
        start_index = -1
        max_start_index = -1

        for i, value in enumerate(self[row_index]):
            if value == 1:
                if current_count == 0:  # New streak starts
                    start_index = i
                current_count += 1
            else:
                if current_count > max_count:  # Update max streak info
                    max_count = current_count
                    max_start_index = start_index
                current_count = 0  # Reset streak

        # Final check in case the longest streak ends at the last element
        if current_count > max_count:
            max_count = current_count
            max_start_index = start_index

        return max_count, max_start_index
    
    def traceArtronFlow(self, column_index):
        column_index+=1
        for row_index in range(len(self)):
            if self[row_index][column_index] != 0:
                return row_index
        raise ValueError(f"column {column_index} doesn't contain non-zero elements")
                
    def scanForChronoHarmony(self):
        chronoHarmonicSwirls = []
        for column_index in range (0, len(self)):
            if self.countCol(column_index) == 1:
                chronoHarmonicSwirls.append( (self.traceArtronFlow(column_index), column_index ) )
        return chronoHarmonicSwirls
        
    def __str__(self):
        string = ""
        for index, row in enumerate(self):  # Use enumerate to track index
        #     string += f"{index} | " + str(row[0]) + ":  " + " ".join(map(str, row[1:])) + "\n"

            grouped_row = [" ".join(map(str, row[i:i+4])) for i in range(1, len(row), 4)]

            # Join the groups with " - " between them
            row_str = " - ".join(grouped_row)

            string += f"{index} | {row[0]}:  " + row_str + "\n"
        return string

def EventDensity(counter, duration, priority):
    return counter/duration - priority * (10*planningHorizon)

def longestStreak(string):
    max_count = 0
    current_count = 0
    start_index = -1
    max_start_index = -1

    for i, value in enumerate(string):
        if value == 1:
            if current_count == 0:  # New streak starts
                start_index = i
            current_count += 1
        else:
            if current_count > max_count:  # Update max streak info
                max_count = current_count
                max_start_index = start_index
            current_count = 0  # Reset streak

    # Final check in case the longest streak ends at the last element
    if current_count > max_count:
        max_count = current_count
        max_start_index = start_index

    return max_count, max_start_index
    
class TimeVortex():
    def __init__(self, time_matrix=TimeMatrix([]), rowTimeLocked = -1, generate_data=False):
        if not isinstance(time_matrix, TimeMatrix):
            raise TypeError(f"invalid argument for TimeVortex: can't accept {type(time_matrix).__name__}, only TimeMatrix!")
        self.matrix = time_matrix.copy()
        self.data = {}
        self.rowTimeLocked = rowTimeLocked
        self.isVortexSettled = False
        
        if self.rowTimeLocked >= len(self.matrix):
            self.isVortexSettled = True
            
        template = {"duration": None, "priority": None, "name": "default Event"}

        if generate_data:
            for index, row in enumerate(self.matrix):
                self.data[row[0]] = template.copy()
                streakLength, _ = longestStreak(row)
                
                self.data[row[0]]["duration"] = random.randint(1, streakLength)
                self.matrix.cleanUp(index, self.data[row[0]]["duration"])
                self.data[row[0]]["priority"] = random.randint(1, 5)
        else:
            for row in self.matrix:
                self.data[row[0]] = template.copy()
                
    def copy(self):
        if self.isVortexSettled:
            return self
        copy = TimeVortex(self.matrix, self.rowTimeLocked)
        copy.data = self.data.copy()
        return copy
            
    def add(self, addition, generate_data = True, priority = 1, duration = 3):
        if len(self.matrix) == 0:  # Handle empty matrix case
            self.matrix = TimeMatrix([addition])  # Initialize matrix with first row
            uID = self.matrix[0, 0]  # First row, first column (ID)
            # if generate_data:
            #     streakLength, _ = longestStreak(self.matrix[-1])
            #     duration = random.randint(0, streakLength)
            #     if duration == 0: duration = 1
            #     priority = random.randint(1, 5)
            # self.data[uID] = {"duration": duration, "priority": priority, "isSet": False}
            
        elif len(addition) == len(self.matrix[0])-1:
            self.matrix = self.matrix.add_row(addition)
            uID=self.matrix[-1,0]
        else:
            raise ValueError(f"can't add line {len(addition)} long to matrix {len(self.matrix[0])-1}")
            
        if generate_data:
            streakLength, _ = longestStreak(self.matrix[-1])
            duration = random.randint(0, streakLength)
            if duration == 0: duration = 1
            priority = random.randint(1, 5)
            
        self.data[uID] = {"duration": duration, "priority": priority, "name": "default Event"}
        self.matrix.cleanUp(-1, self.data[uID]["duration"])
        self.isVortexSettled = False
            
        return self

    def getRowByID(self, uid):
        for index, row in enumerate(self.matrix):
            if row[0] == uid:
                return index, row.copy()
    
    def pop(self, index=-1):
        self.matrix, popped_row = self.matrix.pop_row(index)
        if 0 < index < self.rowTimeLocked:
            self.rowTimeLocked = index - 1 
        popped_id = popped_row[0]  # Extract the unique ID
        if popped_id in self.data:
            del self.data[popped_id]  # Remove metadata entry

        return self
    
    def update_name(self, uID, new_name):
        self.data[uID]["name"] = new_name
    
    def density(self, row_index):
        p = self.data [ self.matrix[row_index][0] ] ["priority"] 
        d = self.data [ self.matrix[row_index][0] ] ["duration"] 
        c = self.matrix.countRow(row_index)
        if c == 0:
            return 256
        return EventDensity(c, d, p)
    
    def sortByDensityBelowRow(self, row):
        for i in range (row+1, len(self.matrix)):
            min_density = self.density(i)
            min_density_index = i
            for j in range(i, len(self.matrix)):  # Iterate through rows below `row`
                current_density = self.density(j)
                if current_density < min_density:
                    min_density = current_density
                    min_density_index = j
            self.matrix.surface(min_density_index, i)
        return self


    def cleanUp(self):
        for index in range(self.rowTimeLocked, len(self.matrix)):
            duration = self.data [ self.matrix[index][0] ] ["duration"]
            self.matrix.cleanUp(index, duration)
    
    def scanForChronoHarmony(self):
        chronoHarmonicSwirls = []
        for column_index in range (0, len(self.matrix[0])-1):
            if self.matrix.countCol(column_index) == 1:
                located_row = self.matrix.traceArtronFlow(column_index)
                if located_row > self.rowTimeLocked:
                    chronoHarmonicSwirls.append( (located_row, column_index ) )
        return chronoHarmonicSwirls
    
    def seekContinuumBalance(self, swirls):
        rows = {}
        for x, y in swirls:
            if x not in rows:
                rows[x] = []
            rows[x].append(y)
        for x, y_values in rows.items():
            y_values.sort()
            uID = self.matrix [ x ] [0]
            duration = self.data[uID]["duration"]
            for i in range(len(y_values) - duration + 1):
                # Check if the sequence from y_values[i] to y_values[i + duration - 1] is consecutive
                if all(y_values[j] == y_values[i] + j for j in range(duration)):
                    return x, y_values[i]  # Found a valid sequence
        else: return None, None
    
    def applyContinuumBalance(self):
        if self.rowTimeLocked+1 >= len(self.matrix):
            self.isVortexSettled = True
            return self
        if self.scanForChronoHarmony():
            xv, yv = self.seekContinuumBalance( self.scanForChronoHarmony() )
            if not xv == None:
                # print(f"locking in:{self.matrix[xv][0], xv, yv}")
                self.timeLock(xv, yv)
                return self.applyContinuumBalance()
        else: return self
    
    def formVortexTrail(self, index):
        """find every beginning of the non-zero sequence in the indexed row, return a list"""
        trail = []
        duration = self.data [ self.matrix[index][0] ] ["duration"] 
        for i in range(1, len(self.matrix[index])):
            if canFit(self.matrix[index], i, duration):
                trail.append(i-1)
        return trail
    
    def timeLock(self, row, col):
        uid = self.matrix[row][0]
        duration = self.data[uid]["duration"]
        if self.rowTimeLocked <= len(self.matrix):
            self.rowTimeLocked+=1
        self.matrix.surface(row, self.rowTimeLocked)
        if duration == 1:
            self.matrix.nullifyExcept(self.rowTimeLocked, col)
            self.matrix.nullify(self.rowTimeLocked, col)
            self.cleanUp()
            self.sortByDensityBelowRow(self.rowTimeLocked)
            return self
        self.matrix.nullifyExcept(self.rowTimeLocked, col, col+duration)
        self.matrix.nullify(self.rowTimeLocked, col, col+duration)
        self.cleanUp()
        self.sortByDensityBelowRow(self.rowTimeLocked)
        if self.rowTimeLocked > len(self.matrix):
            self.isVortexSettled = True
        return self
    
    def wibbleWobbleFit(self):
        if self.rowTimeLocked + 1 >= len(self.matrix):
            self.isVortexSettled = True
            return self
        vortex_trail = self.formVortexTrail ( self.rowTimeLocked + 1 )
        if vortex_trail:
            timeline = random.choice(vortex_trail)
            self.timeLock( self.rowTimeLocked + 1, timeline )
        else:
            # print("no swirls?")
            self.isVortexSettled = True
            return self
        
    def doomedThreads(self):
        for row in range(len(self.matrix)):
            if self.matrix.countRow(row):
                continue
            else:
                return len(self.matrix) - row
        return 0
    
    
    def exploreTimeline(self):
        """ return: self, does matrix have dead events """
        while not self.isVortexSettled:
            self.applyContinuumBalance()

            self.wibbleWobbleFit()

        
        if self.rowTimeLocked + 1  >= len (self.matrix):
            return self, 0
        else: return self, self.doomedThreads()
        
            
    
    def plot_matrix(self):
        my_image = []
        if len(self.matrix):
            displaying = self.matrix[:, 1:].copy()
            for i in range (len(displaying)):
                for j in range (5):
                    row = np.array(displaying[i])
                    row[row == 0] = 184
                    my_image.append(row)
            return Image.fromarray(np.array(my_image).astype(np.uint8))
        else: return "void"
    
    def __str__(self):
        string = ""
        if not len(self.matrix):
            return "Mempty"
        for index, row in enumerate(self.matrix):  # Use enumerate to track index
            grouped_row = [" ".join(map(str, map ( lambda x: x * self.data[row[0]]["priority"] , row[i:i+4]) )) for i in range(1, len(row), 4)]
            # Join the groups with " - " between them
            row_str = " - ".join(grouped_row)

            string += f"{index} | {row[0]}:  " + row_str + "\n"


    
        return string
    
def TimeyWimeyPathfinder(Vortex, depth = 0):
    Vortex = Vortex.copy()
    if depth > 10:
        return "NotFound"
    Vortex.applyContinuumBalance()
    if Vortex.isVortexSettled:
        # print ("settled:", Vortex)
        return Vortex
    swirls = Vortex.formVortexTrail( Vortex.rowTimeLocked + 1 )
    if swirls:
        # print(f"swirls exist for {Vortex.rowTimeLocked+1}")
        bestUV, bestV, bestDooms = None, None, len(Vortex.matrix)

        random.shuffle(swirls)

        for swirl in swirls: # pick random swirl from swirls
            print(f"swirl - {swirl}")
            unstableVortex = Vortex.copy()
            unstableVortex.timeLock(unstableVortex.rowTimeLocked+1, swirl)
            hold = unstableVortex.copy()
            # display(hold.plot_matrix())
            vortex, dooms = unstableVortex.exploreTimeline()
            # display(vortex.plot_matrix())
            # print ("dead:", dooms)
            if not dooms:
                return vortex
            if dooms <= bestDooms:
                bestUV = hold
                bestV = vortex
                bestDooms = dooms
    else:
        return Vortex
    if depth == 10:
        return bestV
    return TimeyWimeyPathfinder(bestUV, depth+1)

NUM_BITS = len(baseLine)
CELL_SIZE = 20

class BinaryRowEditor:
    LEFT_MARGIN = 60
    TOP_MARGIN = 20
    HOUR_BORDER_COLOR = "#003B6F"
    HOUR_BORDER_WIDTH = 5

    def __init__(self, root):
        num_rows = (NUM_BITS // timeslotsInDay) + (1 if NUM_BITS % timeslotsInDay else 0)
        canvas_width = timeslotsInDay * CELL_SIZE + self.LEFT_MARGIN
        canvas_height = num_rows * CELL_SIZE + self.TOP_MARGIN

        self.canvas = tk.Canvas(root, 
                                width=canvas_width, 
                                height=canvas_height)
        self.canvas.pack()
        self.state = [0] * NUM_BITS
        self.rects = []
        self.timeslotsInDay = timeslotsInDay

        self.fill_mode = None
        self.already_updated = set()

        # Создаем метки дней
        for row in range(num_rows):
            x = self.LEFT_MARGIN - 10
            y = self.TOP_MARGIN + row * CELL_SIZE + CELL_SIZE // 2
            self.canvas.create_text(x, y, text=f"День {row+1}", anchor="e")

        # Создаем часовые метки
        for col in range(0, timeslotsInDay, 2):
            hour = col // 2
            time_str = f"{hour:02d}:00"
            x = self.LEFT_MARGIN + col * CELL_SIZE + CELL_SIZE - 10
            y = self.TOP_MARGIN - 5
            self.canvas.create_text(x, y, text=time_str, anchor="s", font=("Arial", 8))

        # Создаем ячейки
        for i in range(NUM_BITS):
            row_idx = i // timeslotsInDay
            col_idx = i % timeslotsInDay
            x0 = self.LEFT_MARGIN + col_idx * CELL_SIZE
            y0 = self.TOP_MARGIN + row_idx * CELL_SIZE
            x1 = x0 + CELL_SIZE
            y1 = y0 + CELL_SIZE
            rect = self.canvas.create_rectangle(x0, y0, x1, y1, 
                                                fill="white", outline="black")
            self.rects.append(rect)
        
        for col in range(0, self.timeslotsInDay + 1, 2):
            x = self.LEFT_MARGIN + col * CELL_SIZE
            self.canvas.create_line(
                x, self.TOP_MARGIN - 1, 
                x, self.TOP_MARGIN + num_rows * CELL_SIZE,
                fill=self.HOUR_BORDER_COLOR,
                width=self.HOUR_BORDER_WIDTH,
                tag="hour_border"
            )

        self.canvas.tag_raise("hour_border")
            
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def toggle_bit_at(self, x, y, initial=False):
        adj_x = x - self.LEFT_MARGIN
        adj_y = y - self.TOP_MARGIN

        if adj_x < 0 or adj_y < 0:
            return

        col = adj_x // CELL_SIZE
        row = adj_y // CELL_SIZE

        if col < 0 or col >= self.timeslotsInDay:
            return
        if row < 0 or row >= (NUM_BITS // self.timeslotsInDay + (1 if NUM_BITS % self.timeslotsInDay else 0)):
            return

        index = row * self.timeslotsInDay + col
        if 0 <= index < NUM_BITS:
            current = self.state[index]

            if initial:
                self.fill_mode = bool(1 - current)

            if index in self.already_updated:
                return

            if self.state[index] != self.fill_mode:
                self.state[index] = self.fill_mode
                self.canvas.itemconfig(self.rects[index], fill="black" if self.state[index] else "white")
                self.already_updated.add(index)

    def on_click(self, event):
        self.already_updated.clear()
        self.toggle_bit_at(event.x, event.y, initial=True)

    def on_drag(self, event):
        self.toggle_bit_at(event.x, event.y)

    def on_release(self, event):
        self.fill_mode = None
        self.already_updated.clear()

class TardisUI:
    PRIORITY_OPTIONS = {
        1: "Совсем неважно",
        2: "Не очень важно",
        3: "Обычное событие",
        4: "Довольно важно",
        5: "Очень важно"
    }

    def __init__(self, root, vortex):
        self.root = root
        self.vortex = vortex
        self.open_editors = {}
        self.current_schedule = None
        
        self.create_scrollable_frame() 
        self.build_main_interface()

    def create_scrollable_frame(self):
        self.main_frame = tk.Frame(self.root, bg="#003b6f")
        self.main_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.main_frame, bg="#003b6f", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#003b6f")

        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    
    def build_main_interface(self):
        self.root.title("События")
        
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        controls_frame = tk.Frame(self.scrollable_frame, bg="#003b6f")
        controls_frame.grid(row=0, column=0, sticky="ew", pady=10)

        left_buttons = tk.Frame(controls_frame, bg="#003b6f")
        left_buttons.pack(side="top", padx=10, pady=2)
        
        add_button = tk.Button(left_buttons, text="+ Новое событие", 
                             command=self.add_event_dialog)
        add_button.pack(side="left", padx=2)
        
        schedule_btn = tk.Button(left_buttons, text="Показать расписание", 
                               command=self.show_schedule)
        schedule_btn.pack(side="left", padx=2)
        
        upd_btn = tk.Button(left_buttons, text="Обновить расписание", 
                          command=self.update_schedule)
        upd_btn.pack(side="left", padx=2)
        
        center_buttons = tk.Frame(controls_frame, bg="#003b6f")
        center_buttons.pack(side="right", padx=10, pady=2)
        
        save_btn = tk.Button(center_buttons, text="Экспорт", 
                           command=self.save_schedule)
        save_btn.pack(side="left", padx=2)
        
        
        right_buttons = tk.Frame(controls_frame, bg="#003b6f")
        right_buttons.pack(side="right", padx=10, pady=20)
        
        vsave_btn = tk.Button(right_buttons, text="Сохранить проект", 
                            command=self.save_vortex)
        vsave_btn.pack(side="left", padx=2)
        
        vload_btn = tk.Button(right_buttons, text="Загрузить проект", 
                            command=self.load_vortex)
        vload_btn.pack(side="left", padx=2)
        
        # Настройка растягивания для основного фрейма
        controls_frame.columnconfigure(0, weight=1)
        controls_frame.columnconfigure(1, weight=1)
        controls_frame.columnconfigure(2, weight=1)

        for index, row in enumerate(self.vortex.matrix, start=1):
            event_id = row[0]
            event_data = self.vortex.data[event_id]
            self.create_event_card(index, event_id, event_data)
            

    def save_vortex(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle файлы", "*.pkl"), ("Все файлы", "*.*")]
        )
        if not file_path:
            return
    
        try:
            data_to_save = {
                'vortex': self.vortex,
                'schedule': self.current_schedule
            }
            with open(file_path, 'wb') as f:
                pickle.dump(data_to_save, f)
            messagebox.showinfo("Успех", "Проект успешно сохранён!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {str(e)}")

    def load_vortex(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Pickle файлы", "*.pkl"), ("Все файлы", "*.*")]
        )
        if not file_path:
            return
    
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if 'vortex' not in data or not isinstance(data['vortex'], TimeVortex):
                raise TypeError("Некорректный формат файла проекта")
            
            self.vortex = data['vortex']
            self.current_schedule = data.get('schedule', None)
            self.refresh_ui()
            messagebox.showinfo("Успех", "Проект успешно загружен!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {str(e)}")

    def save_schedule(self):
        if not self.current_schedule:
            messagebox.showwarning("Предупреждение", "Нет расписания для сохранения.")
            return
    
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )
        if not file_path:
            return
    
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for day_data in self.current_schedule:
                    f.write(f"=== День {day_data['day']} ===\n")
                    for event in day_data['events']:
                        intervals = ", ".join(event["intervals"])
                        priority = self.PRIORITY_OPTIONS[event["priority"]]
                        f.write(f"• {event['name']}\n")
                        f.write(f"  Приоритет: {priority}\n")
                        f.write(f"  Время: {intervals}\n\n")
            messagebox.showinfo("Успех", "Расписание экспортировано в текстовый файл!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при экспорте: {str(e)}")

    def load_schedule(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Pickle файлы", "*.pkl"), ("Все файлы", "*.*")]
        )
        if not file_path:
            return

        try:
            with open(file_path, 'rb') as f:
                schedule = pickle.load(f)
            self.current_schedule = schedule
            self.refresh_ui()
            messagebox.showinfo("Успех", "Расписание успешно загружено!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {str(e)}")

    def show_schedule(self):
        if not self.current_schedule:
            self.current_schedule = self.generate_schedule()
        schedule = self.current_schedule

        window = tk.Toplevel(self.root)
        window.title("Ваше расписание")
        window.geometry("1000x800")

        style = ttk.Style()
        style.configure("Treeview.Heading", font=("Arial", 12, "bold"))
        style.configure("Treeview", rowheight=30, font=("Arial", 11))
        style.configure("Day.Header", font=("Arial", 14, "bold"), background="#e0e0e0")

        tree = ttk.Treeview(window, columns=("time", "event", "priority"), show="headings")
        tree.heading("time", text="Время", anchor="w")
        tree.heading("event", text="Событие", anchor="w")
        tree.heading("priority", text="Приоритет", anchor="w")
        tree.column("time", width=150)
        tree.column("event", width=600)
        tree.column("priority", width=200)

        tree.tag_configure('day_header', background='#f0f0f0', font=('Arial', 12, 'bold'))

        current_day = None
        for day_data in schedule:
            tree.insert("", "end", 
                      values=(f"День {day_data['day']}", "", ""),
                      tags=('day_header',))

            for event in day_data['events']:
                for interval in event["intervals"]:
                    tree.insert("", "end", 
                               values=(interval, event["name"], 
                                       self.PRIORITY_OPTIONS[event["priority"]]),
                               tags=(f"Priority{event['priority']}",))

        scrollbar = ttk.Scrollbar(window, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def generate_schedule(self):
        copy = self.vortex.copy()
        timeline = TimeyWimeyPathfinder(copy)

        def slot_to_time(slot_idx):
            total_minutes = (slot_idx % timeslotsInDay) * timeslotLength
            hours = total_minutes // 60
            minutes = total_minutes % 60
            return f"{hours:02d}:{minutes:02d}"

        schedule_dict = defaultdict(lambda: {'day': 0, 'events': []})

        for row in timeline.matrix:
            event_id = row[0]
            event_data = timeline.data[event_id]
            bits = row[1:]

            day_events = defaultdict(list)
            current_start = None
            current_day = 0

            for i, bit in enumerate(bits):
                day = (i // timeslotsInDay) + 1
                if bit == 1:
                    if current_start is None:
                        current_start = i
                        current_day = day
                    elif day != current_day:
                        start_time = slot_to_time(current_start)
                        end_time = slot_to_time(i)
                        day_events[current_day].append(f"{start_time}-{end_time}")
                        current_start = i
                        current_day = day
                elif current_start is not None:
                    start_time = slot_to_time(current_start)
                    end_time = slot_to_time(i)
                    day_events[current_day].append(f"{start_time}-{end_time}")
                    current_start = None

            if current_start is not None:
                start_time = slot_to_time(current_start)
                end_time = slot_to_time(len(bits))
                day_events[current_day].append(f"{start_time}-{end_time}")

            for day, intervals in day_events.items():
                schedule_dict[day]['events'].append({
                    "name": event_data["name"],
                    "priority": event_data["priority"],
                    "intervals": intervals
                })
                schedule_dict[day]['day'] = day

        sorted_schedule = sorted(schedule_dict.values(), key=lambda x: x['day'])
        for day in sorted_schedule:
            day['events'] = sorted(day['events'], 
                                 key=lambda x: x['intervals'][0] if x['intervals'] else "")

        return sorted_schedule

    def update_schedule(self):
        self.current_schedule = self.generate_schedule()
        self.show_schedule()
    
        
    def create_event_card(self, row, event_id, event_data):
        priority_colors = {
            1: "#d1e7dd",
            2: "#bee3f8",
            3: "#fefcbf",
            4: "#fde68a",
            5: "#fca5a5"
        }

        frame = tk.Frame(self.scrollable_frame, bd=2, relief="raised", bg=priority_colors.get(event_data["priority"], "#e5e5e5"))
        frame.grid(row=row, column=0, padx=10, pady=6, sticky="ew")

        controls_frame = tk.Frame(frame, bg=frame["bg"])
        controls_frame.pack(anchor="e", fill="x", expand=True)

        delete_btn = tk.Button(controls_frame, text="×", 
                               command=lambda eid=event_id: self.delete_event(eid),
                               font=("Arial", 14, "bold"), 
                               fg="red", borderwidth=0)
        delete_btn.pack(side="right", padx=5)

        name_label = tk.Label(frame, text=event_data["name"], font=("Arial", 14, "bold"), bg=frame["bg"])
        name_label.pack(anchor="w", padx=10, pady=2)
        name_label.bind("<Button-1>", lambda e: self.edit_event(event_id))

        priority_text = self.PRIORITY_OPTIONS.get(event_data["priority"], "Неизвестный приоритет")
        detail_text = f"Длительность: {event_data['duration'] * 30} мин | Приоритет: {priority_text}"
        detail_label = tk.Label(frame, text=detail_text, font=("Arial", 10), bg=frame["bg"])
        

        detail_label.pack(anchor="w", padx=10, pady=2)
        detail_label.bind("<Button-1>", lambda e: self.edit_event(event_id))

    def edit_event(self, event_id):
        if event_id in self.open_editors:
            self.open_editors[event_id].lift()
            return

        data = self.vortex.data[event_id]
        row_index, row = self.vortex.getRowByID(event_id)
        row_data = row[1:]

        editor = tk.Toplevel(self.root)
        editor.title("Редактировать событие")
        self.open_editors[event_id] = editor

        def on_close():
            del self.open_editors[event_id]
            editor.destroy()

        editor.protocol("WM_DELETE_WINDOW", on_close)

        tk.Label(editor, text="Название:").grid(row=0, column=0)
        name_var = tk.StringVar(value=data["name"])
        tk.Entry(editor, textvariable=name_var).grid(row=0, column=1)

        tk.Label(editor, text="Длительность (минуты):").grid(row=1, column=0)
        duration_var = tk.IntVar(value=data["duration"] * 30)  
        tk.Entry(editor, textvariable=duration_var).grid(row=1, column=1)

        duration_label = tk.Label(editor, text="")
        duration_label.grid(row=1, column=2, padx=5)
        
        def update_duration_label(*args):
            try:
                minutes = int(duration_var.get())
                hours = minutes // 60
                mins = minutes % 60
                parts = []
                if hours > 0:
                    parts.append(f"{hours} ч.")
                if mins > 0 or hours == 0:
                    parts.append(f"{mins} мин.")
                duration_label.config(text=" ".join(parts))
            except:
                duration_label.config(text="Некорректное значение")
        
        # Инициализация и привязка
        duration_var.trace_add("write", update_duration_label)
        update_duration_label()  # Первоначальное обновление


        tk.Label(editor, text="Приоритет:").grid(row=2, column=0)
        current_priority = data.get("priority", 3)
        current_priority_text = self.PRIORITY_OPTIONS.get(current_priority, "")
        priority_var = tk.StringVar(value=current_priority_text)
        priority_combo = ttk.Combobox(editor, textvariable=priority_var, 
                                     values=list(self.PRIORITY_OPTIONS.values()), 
                                     state="readonly")
        priority_combo.grid(row=2, column=1)

        tk.Label(editor, text="Вектор доступности:").grid(row=3, column=0, columnspan=2)
        bin_frame = tk.Frame(editor)
        bin_frame.grid(row=4, column=0, columnspan=2)

        class BinaryEditorWrapper(BinaryRowEditor):
            def __init__(self, root, initial_state):
                super().__init__(root)
                self.state = initial_state.copy()
                for i, bit in enumerate(self.state):
                    self.canvas.itemconfig(self.rects[i], fill="black" if bit else "white")

        binary_editor = BinaryEditorWrapper(bin_frame, row_data)

        def save():
            try:
                selected_priority = next(k for k, v in self.PRIORITY_OPTIONS.items() if v == priority_var.get())
                
                self.vortex.data[event_id]["name"] = name_var.get()
                minutes = int(duration_var.get())
                if minutes <= 0:
                    raise ValueError("Длительность должна быть положительной")
                slots = math.ceil(minutes / 30)
                
                self.vortex.data[event_id]["duration"] = slots
                self.vortex.data[event_id]["priority"] = selected_priority
                self.vortex.matrix[row_index][1:] = binary_editor.state
                self.vortex.cleanUp()
                on_close()
                self.refresh_ui()
            except ValueError:
                messagebox.showerror("Ошибка", "Проверьте корректность введенных данных")
            except StopIteration:
                messagebox.showerror("Ошибка", "Выберите корректный приоритет")

        tk.Button(editor, text="Сохранить", command=save).grid(row=5, column=0, columnspan=2, pady=10)

    def add_event_dialog(self):
        editor = tk.Toplevel(self.root)
        editor.title("Новое событие")

        tk.Label(editor, text="Название:").grid(row=0, column=0)
        name_var = tk.StringVar(value="Новое событие")
        tk.Entry(editor, textvariable=name_var).grid(row=0, column=1)

        tk.Label(editor, text="Длительность (минуты):").grid(row=1, column=0)
        duration_var = tk.IntVar(value=30)  # Значение по умолчанию 30 минут
        tk.Entry(editor, textvariable=duration_var).grid(row=1, column=1)

        duration_label = tk.Label(editor, text="")
        duration_label.grid(row=1, column=2, padx=5)
        def update_duration_label(*args):
            try:
                minutes = int(duration_var.get())
                hours = minutes // 60
                mins = minutes % 60
                parts = []
                if hours > 0:
                    parts.append(f"{hours} ч.")
                if mins > 0 or hours == 0:
                    parts.append(f"{mins} мин.")
                duration_label.config(text=" ".join(parts))
            except:
                duration_label.config(text="Некорректное значение")

        duration_var.trace_add("write", update_duration_label)
        update_duration_label()

        
        tk.Label(editor, text="Приоритет:").grid(row=2, column=0)
        priority_var = tk.StringVar()
        priority_combo = ttk.Combobox(editor, textvariable=priority_var, 
                                    values=list(self.PRIORITY_OPTIONS.values()), 
                                    state="readonly")
        priority_combo.grid(row=2, column=1)
        priority_combo.set(self.PRIORITY_OPTIONS[3])

        tk.Label(editor, text="Вектор доступности:").grid(row=3, column=0, columnspan=2)
        bin_frame = tk.Frame(editor)
        bin_frame.grid(row=4, column=0, columnspan=2)

        class BinaryEditorWrapper(BinaryRowEditor):
            def __init__(self, root, initial_state=None):
                super().__init__(root)
                if initial_state:
                    self.state = initial_state.copy()
                    for i, bit in enumerate(self.state):
                        self.canvas.itemconfig(self.rects[i], fill="black" if bit else "white")

        binary_editor = BinaryEditorWrapper(bin_frame, [0]*NUM_BITS)

        def save_new_event():
            try:
                minutes = int(duration_var.get())
                if minutes <= 0:
                    raise ValueError("Длительность должна быть положительной")
                slots = math.ceil(minutes / 30)
                
                selected_priority = next(k for k, v in self.PRIORITY_OPTIONS.items() if v == priority_var.get())
                
                new_row = binary_editor.state
                self.vortex.add(new_row, generate_data=False)
                new_id = self.vortex.matrix[-1][0]
                self.vortex.data[new_id] = {
                    "name": name_var.get(),
                    "duration": slots,
                    "priority": selected_priority
                }
                self.vortex.cleanUp()
                editor.destroy()
                self.refresh_ui()
            except ValueError:
                messagebox.showerror("Ошибка", "Проверьте корректность введенных данных")
            except StopIteration:
                messagebox.showerror("Ошибка", "Выберите корректный приоритет")

        tk.Button(editor, text="Создать", command=save_new_event).grid(row=5, column=0, columnspan=2, pady=10)

    def delete_event(self, event_id):
        if messagebox.askyesno("Подтверждение", "Удалить это событие?"):
            for idx, row in enumerate(self.vortex.matrix):
                if row[0] == event_id:
                    self.vortex.matrix = self.vortex.matrix.kill_row(idx)
                    del self.vortex.data[event_id]
                    self.refresh_ui()
                    break

    def refresh_ui(self):
        self.build_main_interface()

if __name__ == "__main__":


    # Инициализация приложения
    vortex = TimeVortex(TimeMatrix([]))
    root = tk.Tk()
    root.geometry("445x600")
    # root.iconbitmap('appicon.ico')
    app = TardisUI(root, vortex)
    root.mainloop()
    
