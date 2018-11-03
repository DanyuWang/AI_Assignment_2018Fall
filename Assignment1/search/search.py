# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import copy


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """*** 11611512 ***"""
    from util import Stack

    stack = Stack()
    visited = []
 
    if problem.isGoalState(problem.getStartState()):
        return []
    else:
        stack.push((problem.getStartState(), []))

        while not stack.isEmpty():
            current = stack.pop()

            if current[0] not in visited:
                if problem.isGoalState(current[0]):
                    return current[1]
                neighbor = problem.getSuccessors(current[0])
                visited.append(current[0])

                for node, step, cost in neighbor:
                    if node not in visited:
                        stack.push((node, current[1] + [step]))

    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """*** 11611512 ***"""
    from util import Queue

    queue = Queue()
    visited = []
    if problem.isGoalState(problem.getStartState()):
        return []
    else:
        queue.push((problem.getStartState(), []))

        while not queue.isEmpty():
            current = queue.pop()

            if current[0] not in visited:
                if problem.isGoalState(current[0]):
                    return current[1]

                neighbor = problem.getSuccessors(current[0])
                visited.append(current[0])

                for node, stage, cost in neighbor:
                    if node not in visited:
                       queue.push((node, current[1] + [stage]))

    util.raiseNotDefined()


def uniformCostSearch(problem):
    "*** 11611512 ***"
    from util import PriorityQueue

    visited = []
    pq = PriorityQueue()

    if problem.isGoalState(problem.getStartState()):
        return []

    else:
        pq.push((problem.getStartState(), []), 0)

        while not pq.isEmpty():
            current = pq.pop()
            if current[0] not in visited:
                visited.append(current[0])

                if (problem.isGoalState(current[0])):
                    return current[1]
                for side, direction, cost in problem.getSuccessors(current[0]):
                    final = current[1] + [direction]
                    pq.push((side, final), problem.getCostOfActions(final))


    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    "*** 11611512 ***"
    from util import PriorityQueue

    open_list = PriorityQueue()
    visited = []
    if problem.isGoalState(problem.getStartState()):
        return []
    else:
        open_list.push((problem.getStartState(),[]),0)
        while not open_list.isEmpty():
            current = open_list.pop()
            if problem.isGoalState(current[0]):
                return current[1]

            if current[0] not in visited:
                visited.append(current[0])
                neighbor = problem.getSuccessors(current[0])

                for side, direction, cost in neighbor:
                    newPath = current[1]+[direction]
                    newCost = problem.getCostOfActions(newPath) + heuristic(side, problem)
                    if side not in visited:
                        open_list.push((side, newPath), newCost)


    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
