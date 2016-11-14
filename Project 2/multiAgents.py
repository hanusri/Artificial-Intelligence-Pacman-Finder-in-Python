# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        #calculate based on ghost
        ghostStatus = 0
        for ghostId in range(len(newGhostStates)):
            dist = util.manhattanDistance(newPos, newGhostStates[ghostId].getPosition())

            if dist == 0:
                return -100

            if newScaredTimes[ghostId] == 0:
                if dist <= 3:
                    ghostStatus -= (1. / dist) * 100

            elif newScaredTimes[ghostId] >= 35:
                ghostStatus += (2. / dist) * 100

            elif newScaredTimes[ghostId] >= 10:
                ghostStatus += (1. / dist) * 100
        
        #calculate based on food
        nearestFood = self.getClosestDistance(newPos, newFood)
        if nearestFood <= 5:
            nearestFood = 30
        else:
            nearestFood = 0;
        if newPos in currentGameState.getFood().asList():
            nearestFood += 200
        # returning summation of food component and ghost component
        return nearestFood + ghostStatus

    def getClosestDistance(self, position, objects):
        if len(objects) == 0:
            return 0
        return min([manhattanDistance(position, obj) for obj in objects])


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(fe):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        return self.minMaxValue(gameState, 0)[0]

    def minMaxValue(self, gameState, depth):
        if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return (None, self.evaluationFunction(gameState))
        #pacman is in zeroth position of number agents.
        # When depth is multiple of total number agents, it is turn for max function as it is pacman's turn to play
        return self.calculateMinMax(gameState, depth, (depth % gameState.getNumAgents() == 0))

    def calculateMinMax(self, gameState, depth, maxFlag):
        # when number agent is zero max function is executed.
        if maxFlag:
            actions = gameState.getLegalActions(0)
            if len(actions) == 0:
                return (None, self.evaluationFunction(gameState))

            maxValue = (None, -float("inf"))
            for action in actions:
                succ = gameState.generateSuccessor(0, action)
                res = self.minMaxValue(succ, depth + 1)
                if res[1] > maxValue[1]:
                    maxValue = (action, res[1])
            return maxValue
        else:
            actions = gameState.getLegalActions(depth % gameState.getNumAgents())
            if len(actions) == 0:
                return (None, self.evaluationFunction(gameState))

            minValue = (None, float("inf"))
            for action in actions:
                succ = gameState.generateSuccessor(depth % gameState.getNumAgents(), action)
                res = self.minMaxValue(succ, depth + 1)
                #return the min value as mentioned in algorithm
                if res[1] < minValue[1]:
                    minValue = (action, res[1])
            return minValue


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.alphaBetaValue(gameState, -float("inf"), float("inf"), 0)[0]

    def alphaBetaValue(self, gameState, alpha, beta, depth):
        if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return (None, self.evaluationFunction(gameState))
        return self.calculateAlphaBeta(gameState, alpha, beta, depth, (depth % gameState.getNumAgents() == 0))

    def calculateAlphaBeta(self, gameState, alpha, beta, depth, maxFlag):
        # when number agent is zero max function is executed.
        if maxFlag:
            actions = gameState.getLegalActions(0)
            if len(actions) == 0:
                return (None, self.evaluationFunction(gameState))

            maxValue = (None, -float("inf"))
            for action in actions:
                succ = gameState.generateSuccessor(0, action)
                res = self.alphaBetaValue(succ,alpha,beta, depth + 1)
                if res[1] > maxValue[1]:
                    maxValue = (action, res[1])
                if maxValue[1] > beta:
                    return maxValue
                alpha = max(alpha, maxValue[1])
            return maxValue
        else:
            actions = gameState.getLegalActions(depth % gameState.getNumAgents())
            if len(actions) == 0:
                return (None, self.evaluationFunction(gameState))

            minValue = (None, float("inf"))
            for action in actions:
                succ = gameState.generateSuccessor(depth % gameState.getNumAgents(), action)
                res = self.alphaBetaValue(succ,alpha,beta, depth + 1)
                if res[1] < minValue[1]:
                    minValue = (action, res[1])
                if minValue[1] < alpha:
                    return minValue
                beta = min(beta, minValue[1])
            return minValue


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.expectiMaxValue(gameState, 0)[0]

    def expectiMaxValue(self, gameState, depth):
        if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return (None, self.evaluationFunction(gameState))
        return self.calculateExpectiMax(gameState, depth, (depth % gameState.getNumAgents() == 0))

    def calculateExpectiMax(self, gameState, depth, maxFlag):
        # when number agent is zero max function is executed.
        if maxFlag:
            actions = gameState.getLegalActions(0)
            if len(actions) == 0:
                return (None, self.evaluationFunction(gameState))

            maxValue = (None, -float("inf"))
            for action in actions:
                succ = gameState.generateSuccessor(0, action)
                res = self.expectiMaxValue(succ, depth + 1)
                if res[1] > maxValue[1]:
                    maxValue = (action, res[1])
            return maxValue
        else:
            actions = gameState.getLegalActions(depth % gameState.getNumAgents())
            if len(actions) == 0:
                return (None, self.evaluationFunction(gameState))
            probability = 1. / len(actions)
            minValue = 0
            for action in actions:
                succ = gameState.generateSuccessor(depth % gameState.getNumAgents(), action)
                res = self.expectiMaxValue(succ, depth + 1)
                minValue += res[1] * probability
            return (None,minValue)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    score = currentGameState.getScore()
    if currentGameState.isWin() or currentGameState.isLose():
        return score

    # calculate based on ghost
    ghostEvaluation = 0
    for ghostId in range(len(newGhostStates)):
        dist = util.manhattanDistance(newPos, newGhostStates[ghostId].getPosition())

        if dist == 0:
            return -100

        if newScaredTimes[ghostId] == 0:
            if dist <= 3:
                ghostEvaluation -= (1. / dist) * 100

        elif newScaredTimes[ghostId] >= 35:
            ghostEvaluation += (2. / dist) * 100

        elif newScaredTimes[ghostId] >= 10:
            ghostEvaluation += (1. / dist) * 100

    # calculate based on food
    nearestFood = getClosestDistance(newPos, newFood)

    return score - 5 * (currentGameState.getNumFood() - ghostEvaluation - 1.5 / nearestFood)


def getClosestDistance(position, objects):
    if len(objects) == 0:
        return 0
    return min([manhattanDistance(position, obj) for obj in objects])


# Abbreviation
better = betterEvaluationFunction
