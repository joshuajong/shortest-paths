"""
Student ID: 31190863

This module contains functions for Task 1 and Task 2. The Graph, Vertex and Edge classes are used for
both tasks.

"""
# Import inf from math
from math import inf
# MinHeap class
"""
Code for MinHeap class is adapted from code for Heap class in FIT1008
"""
class MinHeap:
    def __init__(self, size):
        """
        Initialise the heap. Index array holds the index of the vertices. For example,
        self.index[2] = 3 means Vertex 2 is at index 3 in the vertices array. Length
        refers to the number of vertices (ie len(vertices)+1).
        :param size: number of vertices in the graph 
        :time complexity: best & worst: O(size) to create the index array
        :aux space complexity: best & worst: O(size) for index array
        """
        self.length = 0
        self.vertices = [None]               # root of heap starts at 1
        self.index = [None]*(size+1)         

    def add(self, vertex): 
        """
        Adds a vertex to the heap by appending it to the end of the heap array then
        making it rise to the corrrect position.
        :param vertex: vertex to be added into the heap
        :time complexity: best: O(1) where the vertex is already at the correct position
                                without needing to rise
                          worst: O(log V) where V is the number of vertices in the heap
                                 since the vertex needs to rise
        :aux space complexity: best & worst: O(1)
        """
        self.length = self.length + 1
        self.vertices.append(vertex)
        self.index[vertex.id] = self.length       # update index array
        self.rise(self.length)

    def get_min(self):
        """
        Return the mimimum vertex (the root) of the heap and maintains min heap invariants.
        :return: root of the heap or None if heap is empty
        :time complexity: best & worst: depends on complexity of sink function
        :time complexity: best & worst: O(1)
        """
        if self.is_empty():
            return None
        elif self.length > 1:
            self.swap(1, self.length)
            min_vertex = self.vertices.pop()       # remove the vertex
            self.length = self.length - 1          # update length 
            if self.length > 1:
                self.sink(1)                       # sink the new vertex at the root
            return min_vertex
        else:       # self.length == 1
            min_vertex = self.vertices.pop()       # remove the vertex
            self.length = self.length - 1          # update length 
            return min_vertex
            
    def is_empty(self):
        """
        Checks if the heap is empty. If it is, return true.
        :return: true if the heap is empty, false if not empty
        :time complexity: best & worst: O(1)
        :aux space complexity: best & worst: O(1)
        """
        if self.length == 0:
            return True
        else:
            return False

    def rise(self, index):
        """
        Rise the vertex in the heap to maintain a min heap.
        :param index: index of the vertex in heap array to be risen
        :time complexity: best: O(1) where vertex is already in correct position
                          worst: O(log V) where V is the number of vertices in the heap
        :aux space complexity: best & worst: O(1)
        """
        while index > 1 and self.vertices[index].cost < self.vertices[index//2].cost:
            self.swap(index, index//2)
            index = index // 2

    def sink(self, parent):
        """
        Sink vertex to correct position to maintain a min heap.
        :param parent: vertex to be sunk
        :time complexity: best: O(1) where vertex is already in correct position
                          worst: O(log V) where V is the number of vertices in the heap
        :aux space complexity: best & worst: O(1)
        """
        while 2*parent <= self.length:
            child = self.smallest_child(parent)
            if self.vertices[parent].cost <= self.vertices[child].cost:
                break
            self.swap(child, parent)
            parent = child

    def smallest_child(self, parent):
        """
        Get the smallest child given the parent vertex.
        :param parent: parent vertex
        :return: smallest child vertex of the parent vertex
        :time complexity: best & worst: O(1)
        :aux space complexity: best & worst: O(1)
        """
        left_child = 2*parent
        right_child = 2*parent+1
        # check if right child exists 
        if left_child == self.length or self.vertices[left_child].cost < self.vertices[right_child].cost:
            return left_child
        else:
            return right_child

    def swap(self, parent, child):
        """
        Swap the parent and child vertex in the vertex array and their indexes in the index array.
        :param parent: parent vertex
        :param child: child vertex
        :time complexity: best & worst: O(1) since only swapping is done
        :aux space complexity: best & worst: O(1)
        """
        self.vertices[parent], self.vertices[child] = self.vertices[child], self.vertices[parent]
        vertex1 = self.vertices[parent]
        vertex2 = self.vertices[child]
        self.index[vertex1.id], self.index[vertex2.id] = self.index[vertex2.id], self.index[vertex1.id]
    
    def update(self, vertex):
        """
        Update position of vertex in the heap after cost has changed.
        Only need to rise as cost should always be updated to a smaller value.
        :param vertex: vertex to be moved
        :time complexity: best & worst: depends on complexity of rise
        :aux space complexity: best & worst: O(1)
        """
        array_index = self.index[vertex.id]
        self.rise(array_index)

    def __len__(self):
        """
        Returns the number of vertices currently in the heap
        :return: length attribute
        :time complexity: best & worst: O(1)
        :aux space complexity: best & worst: O(1)
        """
        return self.length

#  Edge class
"""
Edge class to create edges between vertices in a graph.
"""
class Edge:
    def __init__(self, u, v, w):
        """
        Initialises the vertices who have edges between them and the weight of the edge.
        :param u: vertex u (an integer) 
        :param v: vertex v (an integer)
        :param w: weight of edge (also known as cost in this module)
        :time complexity: best & worst: O(1)
        :aux space complexity: best & worst O(1)
        """
        self.u = u
        self.v = v
        self.w = w
    
    def __str__(self):
        """
        Returns a string tuple in the form of (u,v,w). Used for testing.
        :return: string description
        :time complexity: best & worst: O(1)
        :aux space complexity: best & worst O(1)
        """
        return "(" + str(self.u) + ", " + str(self.v) + ", " + str(self.w) + ")"

# Vertex class
"""
Vertex class to represent a vertex in a graph.
"""
class Vertex:
    def __init__(self, id):
        """
        Initialise attributes stored by a vertex.
        :param id: vertex ID 
        :time complexity: best & worst: O(1)
        :aux space complexity: best & worst O(1)
        """
        self.id = id
        self.edges = []     # list of edges connected to the vertex (adjancency list)
        self.discovered = False     # used for Dijkstra's algorithm
        self.visited = False        # used for Dijkstra's algorithm
        self.previous = None        # used for Dijkstra's algorithm
        self.cost = inf             # used for Dijkstra's and Bellman Ford algorithm
        self.capacity = 0           # used for Bellman Ford algorithm

    def __str__(self):
        """
        Returns the vertex ID. Used for testing.
        :return: string
        :time complexity: best & worst: O(1)
        :aux space complexity: best & worst O(1)
        """
        return str(self.id) 

#Graph class
"""
Graph class for both directed or undirected edges. 
"""
class Graph:
    def __init__(self, v, edges, directed=False):
        """
        Initialise graph with vertices and edges. Edges created will depend on whether
        the graph is directed or undirected.
        :param v: number of vertices
        :param edges: list of edges in the graph
        :param directed: true if graph is directed and false if undirected
        :time complexity: best & worst: O(v) where v is the number of vertices
        :aux space complexity: best & worst: O(v)
        """
        self.vertices = [None]*v        
        for index in range(v):          # create vertices
            self.vertices[index] = Vertex(index)
        for edge in edges:              # create edges
            if directed:
                self.create_edge(edge, True)
            else:                       # undirected edges
                self.create_edge(edge)

    def bellman_ford(self, source, prices, max_trades, edges, trades):
        """
        A modified version of Bellman Ford's algorithm. Instead of looping V-1 times, where
        V is the number of vertices in a graph, the function loops min(max_trades, V-1) times.
        The additional loop to check for negative cycles is also not needed for Task 2 and
        thus omitted in this modified version. Negative cost is stored as finding the maximum
        corresponds to negating all the cost values and finding the minimum. Price of liquid
        and trade ratios are not negative.
        :param source: source vertex
        :param max_trades: maximum number of edges that can be visited in the graph
        :param edges: list of edges in the graph
        :param trades: number of edges already considered (from the previous run of this
                       function – if this is the first time, value will be 0)
        :return: a tuple (trades, profit) where trades is the number of trades performed by the
                 function and profit is a list of length V, where V is the number of vertices in the graph, 
                 containing the profit earned after trading for each vertex 
        :time complexity: best & worst: O(min(max_trades, V)*T) where V is the number of vertices in the  
                          graph and T is the len(edges). In a dense graph (as assumed), T is
                          approximately V^2. Thus,
                          Calculation:
                          O(V) + O(min(max_trades, V)*(T+V)) = O(min(max_trades, V)*T)
        :aux space complexity: best & worst: O(V) for arrays, profit and capacity
        """
        profit = [inf]*len(self.vertices)   # stores profit for the current loop 
        capacity = [0]*len(self.vertices)   # stores capacity for the current loop
        if trades == 0:     # initialise the arrays
            profit[source] = -1*prices[source]
            capacity[source] = 1
            self.vertices[source].cost = -1*prices[source]
            self.vertices[source].capacity = 1
        else:
            # update capacity and profit from previous run of bellman_ford
            for i in range(len(self.vertices)):     # runs in O(V) time
                capacity[i] = self.vertices[i].capacity
                profit[i] = self.vertices[i].cost
        # perform edge relaxation
        for _ in range(len(self.vertices)): # runs in O(min(max_trades, V)) time
            to_change = []        # stores indexes of values to be changed for each time the outer loop runs
            trades = trades + 1         # current loop is trade number trades+1
            if trades > max_trades:     # if exceeded max_trades, terminate loop
                break
            for edge in edges:    # runs in O(T) time
                u = edge[0]
                v = edge[1]
                w = edge[2]
                vertex_v = self.vertices[v]
                vertex_u = self.vertices[u]
                value = -1*vertex_u.capacity*w*prices[v]
                liq_volume = vertex_u.capacity*w
                # update profit and capacity if the profit obtained from the liquid is higher
                if vertex_v.cost > value and value < profit[v]:
                    profit[v] = value
                    capacity[v] = liq_volume
                    to_change.append(v)     # values changed so must keep track for update later
            # update the data in the vertices that should change
            for index in to_change:     # runs in O(V) time
                self.vertices[index].capacity = capacity[index]
                self.vertices[index].cost = profit[index]
        return(trades, profit)
        
    def create_edge(self, edge, directed=False):
        """
        Creates an edge between two vertices by adding the edge into the adjacency lists of
        the vertex (both vertices if edge is undirected).
        :param edge: a tuple (u,v,w) with vertex u, vertex v and weight w
        :param directed: true if the edge is directed and false if undirected
        :time complexity: best & worst: O(1)
        :aux space complexity: best & worst: O(1)
        """
        v1 = edge[0]
        v2 = edge[1]
        cost = edge[2]
        if v1 >= 0 and v1 < len(self.vertices) and v2 >= 0 and v2 < len(self.vertices):
            self.vertices[v1].edges.append(Edge(self.vertices[v1], self.vertices[v2], cost))
            if not directed:
                self.vertices[v2].edges.append(Edge(self.vertices[v2], self.vertices[v1], cost))

    def dijkstra(self, start, destination): 
        """
        Dijkstra's algorithm, modified to suit Task 2.
        :param start: source vertex
        :param initial_cost: initial cost at source vertex (default is 0) - when starting at 
                             the delivery city, the cost will be the profit (a negative integer)
        :param dedstination: destination vertex for possible early termination
        :time complexity: best & worst: O(V^2 log V) which is O(E log V) for dense graphs
                                        where V is the number of vertices and E is the number
                                        of edges in the graph
        :aux space complexity: O(V) due to min heap
        """
        discovered_queue = MinHeap(len(self.vertices))      # create min heap with n vertices
        source = self.vertices[start]              # get the source vertex
        source.discovered = True
        source.cost = 0                            # update source cost with initial cost
        discovered_queue.add(source)

        while not discovered_queue.is_empty():     # runs in O(V) time
            u = discovered_queue.get_min()         # runs in O(log V)
            u.visited = True
            for edge in u.edges:                   # runs in O(V) time
                v = edge.v
                if not v.discovered:
                    v.discovered = True
                    v.previous = u
                    v.cost = u.cost + edge.w       # update cost that was originally inf
                    # add to the heap
                    discovered_queue.add(v)        # runs in O(log V) time
                elif not v.visited:                # if discovered but not visited yet 
                    if v.cost > u.cost + edge.w:   # compare cost and update if necessary
                        v.cost = u.cost + edge.w
                        v.previous = u
                        # update the heap
                        discovered_queue.update(v) # runs in O(log V) time
            if u.id == destination:     # break if already at destination
                break

# Task 1

def best_trades(prices, starting_liquid, max_trades, townspeople):
    """
    Calculates the maximum value of liquid a person can get after trading with various townspeople
    with various liquids. This function appllies a modified version of Bellman Ford's algorithm a
    maximum of max_trades, M times.
    :param prices: list of prices of liquids where liquid i has a price of prices[i]
    :param starting_liquid: liquid person has when arriving in the town (capacity is always 1)
    :param max_trades: maximum number of trades that can be done by the person
    :param townspeople: list of lists where every nested list townspeople[i] contains trades done
                        by townspeople i. Each trade is in the form of 
                        (given_liquid, received_liquid, ratio). The total number of trades is T
    :return: maximum value attained by person after trading various liquids up to M times
    :time complexity: best & worst: O(M*T) due to reasoning in comments below
                                    Calculation: O(T) + O(T) + O(M*T) = O(M*T)
    :aux space complexity: best & worst: O(V) depends on the space taken by the graph and bellman ford 
    """
    if max_trades == 0:
        return prices[starting_liquid]
    edges = []
    for people in townspeople:  # runs in O(T) time
        edges.extend(people)
    # create graph of trades where liquids are vertices and trades are edges
    potential_trades = Graph(len(prices), edges, True)      # runs in O(T) time
    trades = 0      # keep track of number of trades made
    # keep calling bellman_ford if max_trades has not been reached
    # while loop runs in << O(M*T) times so it can be ignored in counting big-O
    vals = (0, [prices[starting_liquid]])
    while trades < max_trades:      # loop runs in O(M*T) time 
        # if bellman_ford is called more than once, then M > V where V is the number of liquids
        # and this loop will run in O(V*T)+O((M-V)*T) = O(M*T)
        # if it is called once, then M <= V in which the function will run in O(M*T) times
        vals = potential_trades.bellman_ford(starting_liquid, prices, max_trades, edges, trades)
        trades_made = vals[0]       # number of trades made in this run of bellman_ford
        trades = trades_made   # update number of trades
    return -1*(min(vals[1]))
    
# Task 2

def opt_delivery(n, roads, start, end, delivery):
    """
    Finds a route from the start to the end city with the minimum cost using Dijkstra's
    algorithm. A delivery may be made to reduce to cost of travelling from the start to 
    end city. 
    :param n: number of cities from 0..n-1, of length N
    :param roads: list of tuples in the form of (u,v,w) where u and v are cities and w is
                  the cost of travelling from u to v or vice versa, of length R
    :param start: ID of start city
    :param end: ID of end city
    :param delivery: tuple (pickup, delivery, profit) where pickup is the city to pickup the
                     item, delivery is the city the item is to be delivered to and profit is
                     the amount made if the delivery is done
    :return: a tuple (cost, route) where cost is the cost of travelling from start to end and
             route is a list of cities in the order it is travelled to from start to end city
    :time complexity: O(R log N) due to running Dijkstra's algorithm. In a connected graph,
                      R is N-1 or O(N) so O(R log N) is more significant than O(N).
                      Calculation: O(N) + O(R log N) + O(R log N) + O(R log N) + O(R) = O(R log N)
    :aux space complexity: aux space complexity of the graph, O(N) for graphs and dijkstra's algorithm
    """
    pickup_city = delivery[0]
    delivery_city = delivery[1]
    profit = delivery[2]

    network = Graph(n, roads)       # create the graph - runs in O(N)
    network.dijkstra(start, pickup_city)         # runs in O(R log N)
    cost_to_pickup = network.vertices[pickup_city].cost     # cost from source to pickup_city

    duplicate1 = Graph(n, roads)             # create duplicate graph and run it from pickup city as source
    duplicate1.dijkstra(pickup_city, delivery_city)         # runs in O(R log N)
    cost_to_delivery = duplicate1.vertices[delivery_city].cost      # cost from pickup city to delivery city

    duplicate2 = Graph(n, roads)             # create duplicate graph and run it from delivery city as source
    duplicate2.dijkstra(delivery_city, end)       # runs in O(R log N)
    cost_to_end = duplicate2.vertices[end].cost     # cost to travel from delivery city to end

    no_delivery = network.vertices[end].cost            # cost of travel without delivery
    with_delivery = cost_to_pickup + cost_to_delivery + cost_to_end - profit        # cost with delivery

    if with_delivery < no_delivery:         # if the delivery is worth it, then take it
        return (with_delivery, build_travel_route(network, start, end, pickup_city, delivery_city, duplicate1, duplicate2))  # runs in O(R) time
    return (no_delivery, build_travel_route(network, start, end))   # runs in O(R) time

def build_travel_route(network, start, end, pickup_city=None, delivery_city=None, duplicate1=None, duplicate2=None):
    """
    Backtracks from the end city back to the first city to build the least costly
    route.
    :param network: graph
    :param start: start city
    :param end: end city
    :param delivery_city: delivery city
    :param duplicate: second graph with source from delivery city
    :return: list of cities showing the route
    :time complexity: best & worst: O(R) due to recursion and list extension
    :aux space complexity: best & worst: O(R) due to the recursion stack 
    """
    if duplicate1 is None:       # backtrack from one graph only
        route = build_travel_route_aux(network, start, end)     # runs in O(R)
        route.append(end)
        return route
    else:
        if delivery_city is not None and duplicate1 is not None and \
            pickup_city is not None and duplicate2 is not None:       # backtrack with a delivery made
            route = build_travel_route_aux(network, start, pickup_city)            # runs in O(R)
            route1 = build_travel_route_aux(duplicate1, pickup_city, delivery_city) # runs in O(R)
            route2 = build_travel_route_aux(duplicate2, delivery_city, end)         # runs in O(R)
            
            route.extend(route1)        # runs in O(R)
            route.extend(route2)        # runs in O(R)
            route.append(end)           
            return route       
        
def build_travel_route_aux(network, start, end):
    """
    Auxilliary function to rebuild the route using recursion.
    :param network: graph
    :param start: start city
    :param end: end city
    :return: list of cities showing the route
    :time complexity: best & worst: O(R) as the recursion depth depends on the number
                                    of roads
    :aux space complexity: best & worst: O(R) due to the recursion stack                                
    """
    end_vertex = network.vertices[end]
    if end_vertex.previous is None:
        return []
    elif end_vertex.previous.id == start:     # base case where the previous city is already the start
        return [end_vertex.previous.id]
    else:
        route = build_travel_route_aux(network, start, end_vertex.previous.id)
        route.append(end_vertex.previous.id)
        return route