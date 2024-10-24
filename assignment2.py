"""
FIT2004 Assignment 2 - A Fun Weekend Away & Spell Checker 

Student Name: YAP HIEW TONG
Student ID: 34568611
Student Email: hyap0021@student.monash.edu

"""


from collections import deque #used in task 1
import re #used in task 2 

class Network_Flow:
    def __init__(self,size):
        """ 
        Function description:  
        This function initializes the Network_Flow object with a given size.
        It creates a residual graph to facilitate the Ford-Fulkerson algorithm.
        Each vertex in the network is represented by a VertexNF object.    

        Approach description:
        - Initialize the residual graph with the given size.
        - Create a list of vertices, each represented by an instance of VertexNF.
        - Populate the vertices list with VertexNF objects.

        Input:
        - size (int): The size of the network flow.

        Precondition:
        - size must be a positive integer.

        Postcondition:
        - A Network_Flow object is created and its vertices and residual graph are initialized.

        Time complexity:
        - Initialising the vertices takes O(n) time, where n is the size of the network flow.
        - Initialising the residual graph also takes O(N) time, where n is the size of the network flow.
        - Total time complexity is O(N), where n is the size of the network flow.

        Auxillary Space complexity:
        - Initialising the vertices requires O(n) space, where n is the size of the network flow.
        - Initialising the residual graph also requires O(N) space, where n is the size of the network flow.
        - Total auxillary space complexity is O(N), where n is the size of the network flow.

        """
        
        self.residual_graph = ResidualGraph(size) 
        self.vertices = [None]*(size)                
        for i in range(len(self.vertices)):
            self.vertices[i] = VertexNF(i)
    
    def add_edge(self,u,v,capacity):
        """
        Function description:
        This function adds an edge to the network flow with a specific capacity,linking it with the residual graph.

        Approach description:
        - Create an EdgeNF object for the forward edge in the network flow.
        - Append this EdgeNF object to the edges list of the source vertex, to allow for efficient traversal of network and find path from source to the sink
        - Add the corresponding forward and reverse edges to the residual graph by calling the add_edge function in ResidualGraph.
        - Link the EdgeNF object in NetworkFlow with the forward edge in ResidualGraph.
        - Linking these edges allows for the flow to be updated and augmented in both the network flow and the residual graph.

        Input:
        - u (int): The source vertex of the edge.
        - v (int): The destination vertex of the edge.
        - capacity (int): The capacity of the edge.

        Precondition:
        - u, v, and capacity must be positive integers.
        - u and v must be within the valid range of vertex IDs.
        - capacity must be a non-negative integer.

        Postcondition:
        - An EdgeNF object is created and added to the edges list of the source vertex.
        - The corresponding forward and reverse edges are added to the residual graph.
        - The EdgeNF object is linked with the forward edge in ResidualGraph.

        Time complexity:
        - O(1) for adding the edge to the network flow.
        - O(1) for adding the corresponding forward and reverse edges to the residual graph.
        - Total time complexity is O(1).

        Auxillary Space complexity:
        - O(1) for creating the EdgeNF object.
        - O(1) for adding the corresponding forward and reverse edges to the residual graph.
        - Total auxillary space complexity is O(1).
        """
        edge_nf = EdgeNF(u,v,capacity)
        self.vertices[u].edges.append(edge_nf)
        #Add corresponding forward and reverse edge to ResidualGraph
        edge_rg = self.residual_graph.add_edge(u,v,capacity)
        # Link the edge in NetworkFlow with the forward edge in ResidualGraph
        edge_nf.residual_edge = edge_rg
        edge_rg.residual_edge = edge_nf
        

    def ford_fulkerson(self,source,sink,preference_count):
        """ 
        Function description:
        This function implements the Ford-Fulkerson algorithm to find the maximum flow in the network.

        Approach description:
        - Initialize max_flow to 0.
        - Continuously find augmenting paths in the residual graph until no more augmenting paths are found.
        - For each augmenting path found, calculate the minimum capacity along the path.
          Update the flow along the path and add it to the total max_flow.
        - Terminate when no more augmenting paths are found.
        - Return the maximum flow.

        Input:
        - source (int): The source vertex of the network.
        - sink (int): The sink vertex of the network.
        - preference_count (list): A list of integers representing the number of preferences for each participant.

        Precondition:
        - The source and sink must be valid vertex IDs within the network.
        - preference_count must be a list of integers with a length equal to the number of participants.

        Postcondition:
        - The maximum flow in the network is calculated and returned.

        Time complexity:
        - The outer while loop will iterate until no more augmenting paths can be found. In the worst-case scenario, the maximum flow will be equal to the total number of participants
          So, the total number of iterations of the outer loop is dependent on the maximum flow, F
        - Each iteration of the outer loop due to BFS traversal takes O(N*M) time, where N is the number of participants and m is the number of activities
        - The time complexity is O(F(N*M)), where F is the maximum flow in the network.
        -Since the maximum flow is bounded by the number of participants, F is less than or equal to N.
        - Therefore, the time complexity is O(N^2M), where N is the number of participants and M is the number of activities.
        - O(N^2*M) also can be written as O(V*E^2)

        Auxillary Space complexity:
        - O(N+M) to store the residual graph and augmenting path information.
        - The augmenting path array has a size of O(N+M)
        - The visited array has a size of O(N+M)
        - The queue used in BFS has a size of O(N+M)
        - Therefore, the total auxillary space complexity is O(N+M)

        """
        
        max_flow = 0
        while True:
            path = self.residual_graph.find_augmenting_path(source,sink,preference_count)
            if path is None:
                break
            min_capacity = float("inf")
            for edge in path:
                 min_capacity = min(min_capacity, edge.residual_capacity)
            self.augment_flow(path,min_capacity)
            
            max_flow += min_capacity
        return max_flow

    def augment_flow(self,path,min_capacity):
        """ 
        Function description:
        This function updates update both the forward and reverse edges in 
        the residual graph to reflect the flow sent along the path.

        Approach description:
        - Iterate through each edge in the path.
        - Call the update_residual_capacity function from class EdgeRG since it's an edge from the ResidualGraph
        - It is to update both the forward and reverse edges in the residual graph.
        - It then checks if the current edge has a corresponding residual edge(forward edge in the network flow), if so, it updates the flow of network flow by the same amount.

        Input:
        - path (list): A list of EdgeRG objects representing the augmenting path.
        - min_capacity (int): The minimum capacity along the augmenting path.

        Precondition:
        - The path must be a valid augmenting path in the residual graph.

        Postcondition:
        - The residual capacities of the edges in the path are updated.
        - The flow of the corresponding edges in the network flow is updated.

        Time complexity:
        - Each edge in the path is processed once to update the residual capacities and flows.
        - The update_residual_capacity function is called for each edge in the path, which takes constant time.
        - Therefore, the time complexity is O(nm), where n is the number of participants and m is the number of activities.

        Auxillary Space complexity:
        - O(1) as it only uses a fixed amount of extra space regardless of the input size.
        """
        for edge in path:
            edge.update_residual_capacity(min_capacity)
            if edge.residual_edge is not None:
                edge.residual_edge.flow += min_capacity
                
class VertexNF:
    def __init__(self,vertex_id):
        """ 
        Function description:
        This function initializes the VertexNF object with a given vertex ID.
        It sets up the vertex's edges list and visited flag.

        Approach description:
        - Initialize the vertex with a given ID.
        - Create an empty list to store the edges.
        - Set the visited flag to False. 

        Input:
        - vertex_id (int): The ID of the vertex.

        Precondition:
        - vertex_id must be a positive integer.

        Postcondition:
        - A VertexNF object is created with the specified ID.
        - The edges list is initialized as empty.
        - The visited flag is set to False.

        Time complexity:
        - O(1) as it only involves initializing the vertex ID, edges list, and visited flag.

        Auxillary Space complexity:
        - O(1) as it only uses a fixed amount of extra space regardless of the input size.

        """
        self.vertex_id = vertex_id
        self.edges = []
        self.visited = False

class EdgeNF:
    def __init__(self,u,v,capacity):
        """ 
        Function description:
        This function initializes the EdgeNF object with a given source vertex, destination vertex, and capacity.
        It also initializes the flow to 0 and sets up the residual edge to link with the residual graph.

        Approach description:
        - Initialize the edge with the given source vertex, destination vertex, and capacity.
        - Set the flow to 0.
        - Initialize the residual edge to None.This is to link with the ResidualGraph later on.
        
        Input:
        - u (int): The source vertex of the edge.
        - v (int): The destination vertex of the edge.
        - capacity (int): The capacity of the edge.

        Precondition:
        - u, v, and capacity must be positive integers.

        Postcondition:
        - An EdgeNF object is created with the specified source, destination, and capacity.
        - The flow is initialized to 0.
        - The residual edge is initialized to None to link with the ResidualGraph later on.

        Time complexity:
        - O(1) as it only involves initializing the edge's source, destination, capacity, flow, and residual edge.

        Auxillary Space complexity:
        - O(1) as it only uses a fixed amount of extra space regardless of the input size.
        
        """
        self.u = u
        self.v = v
        self.capacity = capacity
        self.flow = 0
        self.residual_edge = None #to link with residual graph
    
    
class ResidualGraph:
    def __init__(self,size):
        """ 
        Function description:
        This function initializes the ResidualGraph object with a given size.
        It creates a list of vertices, each represented by an instance of VertexRG.

        Approach description:
        - Initialize the residual graph with the given size.
        - Create a list of vertices, each represented by an instance of VertexRG.
        - Populate the vertices list with VertexRG objects.

        Input:
        - size (int): The size of the residual graph.

        Precondition:
        - size must be a positive integer. 

        Postcondition:
        - A ResidualGraph object is created with the specified size.
        - The vertices list is initialized with VertexRG objects.

        Time complexity:
        - O(N + M) to initialize the vertices list, where N are the number of participants and M are the number of activities
        -Total: O(N), since number of activites will be at most N/2

        Auxillary Space complexity:
        - O(N) to store the vertices list.
        
        """
        self.vertices = [None]*(size)                
        for i in range(len(self.vertices)):
            self.vertices[i] = VertexRG(i)
        

    def add_edge(self,u,v,capacity):
        """ 
        Function description:
        This function adds an edge to the residual graph.

        Approach description:
        - Create an EdgeRG object for the forward edge.(u->v)
        - Create an EdgeRG object for the reverse edge.(v->u)
        - Set the reverse edge's reverse to the forward edge and vice versa.
        - This establishes the bidirectional relationship between the forward and reverse edges, 
          to allow easier access and updates of the reverse edge capacities when flow is adjusted
        - Append the forward edge to the edges list of the start vertex.
        - Append the reverse edge to the edges list of the end vertex.
        - Return the forward edge to link with NetworkFlow.

        Input:
        - u (int): The source vertex of the edge.
        - v (int): The destination vertex of the edge.
        - capacity (int): The capacity of the edge.

        Precondition:
        - u, v, and capacity must be positive integers.

        Return:
        - forward_edge (EdgeRG): The forward edge created in the residual graph.

        Time complexity:
        - O(1) as it only involves creating the forward and reverse edges and appending them to the respective vertices.

        Auxillary Space complexity:
        - O(1) as it only uses a fixed amount of extra space regardless of the input size.
        """
        forward_edge = EdgeRG(u,v,capacity)
        reverse_edge = EdgeRG(v,u,0) 
        #Establish the bidirectional relationship between the forward and reverse edges
        forward_edge.reverse = reverse_edge
        reverse_edge.reverse = forward_edge

        self.vertices[u].edges.append(forward_edge)
        self.vertices[v].edges.append(reverse_edge)

        return forward_edge #to link with Network Flow
    
    def find_augmenting_path(self,source,sink,preference_count):
        """ 
        Function description:
        This function finds an augmenting path from source to sink in the residual graph using BFS,
        prioritising participants with higher preference.

        Approach description:
        - Initialize an array to store the path.
        - Initialize a queue for BFS.
        - Mark all vertices as unvisited to prevent revisiting vertices within the same BFS traversal.
        - Mark the start vertex as visited and enqueue it.
        - Perform BFS starting from the source vertex. 
        - Preference_count represents the number of "preference 2" for each participant.
        - Participant with only one "preference 2" have limited choices, so they should be prioritised by adding them to the front of the queue.
          so that they are considered first in the augmenting path search. Participants with two 'Preference 2" are more flexible as they have more options available.
          This approach ensures that participants with limited options are assigned to their preferred activities first. This maximises the overall number of participants who can be assigned. 
        - While the queue is not empty, dequeue a vertex and check its edges.
        - If there is residual capacity and the vertex hasn't been visited yet, 
          mark it as visited and enqueue it.
        - If the end vertex is reached, return the augmenting path.
        - If no augmenting path is found, return None.

        Input:
        - source (int): The source vertex of the network.
        - sink (int): The sink vertex of the network.
        - preference_count (list): A list of integers representing the number of "preference 2" for each participant.

        Precondition:
        - The source and sink must be valid vertex IDs within the network.
        - preference_count must be a list of integers with a length equal to the number of participants.

        Return:
        - augmenting_path (list): A list of EdgeRG objects representing the augmenting path.

        Time complexity:
        - Traversing the vertices during BFS takes O(n+m),where n is the number of participants and m is the number of activities.
        - Traversing the edges during BFS takes O(nm), as the number of edges connecting participants to activities dominates
        - Therefore, the time complexity is O(N*M) + O(N+M) = O(N*M).
        
        Auxillary Space complexity:
        - The path array has a size equal to the number of vertices, (N+M). Each entry in this array stores an edge, which contains information about the source (u), destination (v), and residual capacity.
          The number of edges stored in the path array is proportional to the number of vertices, (N+M), since the BFS only trace one edge per vertex in the augmenting path.
          Therefore, the space complexity is O(n+m), where N is the number of participants and m is the number of activities.
        - For visited array, it has a size equal to the number of vertices,(N+M) 
          Therefore, the space complexity is O(N+M)
        - The queue is used to store the vertices for BFS traversal. In the worst case, it may store all vertices in the residual graph, so its space complexity is O(n+m).
        - Total auxillary space complexity is O(N+M)
        
        
        """
        #Initialise an array to store the path
        path = [None]*len(self.vertices)
        #Initialise a queue for bfs
        queue = deque([source])
        for vertex in self.vertices:
            vertex.visited = False
        self.vertices[source].visited = True
        while queue:
            u = queue.popleft()
            for edge in self.vertices[u].edges:
                if edge.residual_capacity > 0 and not self.vertices[edge.v].visited:
                    path[edge.v]=edge
                    self.vertices[edge.v].visited = True
                    if edge.v < len(preference_count) and preference_count[edge.v] == 1:
                        queue.appendleft(edge.v)
                    else:
                        queue.append(edge.v)
                    if edge.v == sink:
                        augmenting_path = []
                        current = sink
                        while current != source:
                            edge = path[current]
                            augmenting_path.append(edge)
                            current = edge.u
                        return augmenting_path[::-1]
                    queue.append(edge.v)
        return None

class VertexRG:
    def __init__(self,vertex_id):
        """ 
        Function description:
        This function initializes the VertexRG object with a given vertex ID.
        It sets up the vertex's edges list and visited flag.

        Approach description:
        - Initialize the vertex with a given ID.
        - Create an empty list to store the edges.
        - Set the visited flag to False. 

        Input:
        - vertex_id (int): The ID of the vertex.

        Precondition: 
        - vertex_id must be a positive integer.

        Postcondition:
        - A VertexRG object is created with the specified ID.
        - The edges list is initialized as empty.
        - The visited flag is set to False.

        Time complexity:
        - O(1) as it only involves initializing the vertex ID, edges list, and visited flag. 

        Auxillary Space complexity:
        - O(1) as it only uses a fixed amount of extra space regardless of the input size.
        """
        self.vertex_id = vertex_id
        self.edges = []
        self.visited = False


class EdgeRG:
    def __init__(self,u,v,residual_capacity):
        """ 
        Function description:
        This function initializes the EdgeRG object with a given source vertex, destination vertex, and residual capacity.
        This class is crucial for managing the residual capacities and reverse edges needed for the Ford-Fulkerson algorithm.
        
        Approach description:
        - Initialize the residual capacity, which indicates how much more flow can be sent through this edge.
        - Set up a reference to the reverse edge, which is used to allow flow to be "pushed back" if necessary.
        - Initialize a reference(self.residual_edge) to the corresponding edge in the network flow, allowing for updates to the actual flow in the network. 

        Input:
        - u (int): The source vertex of the edge.
        - v (int): The destination vertex of the edge.
        - residual_capacity (int): The residual capacity of the edge.

        Precondition:
        - u, v, and residual_capacity must be positive integers.

        Postcondition:
        - An EdgeRG object is created with the specified source, destination, and residual capacity.
        - The reverse edge is initialized to None to link with the ResidualGraph later on.
        - The residual edge is initialized to None to link with the NetworkFlow later on.

        Time complexity:
        - O(1) as it only involves initializing the edge's source, destination, residual capacity, reverse edge, and residual edge.

        Auxillary Space complexity:
        - O(1) as it only uses a fixed amount of extra space regardless of the input size.
        """
        self.u = u
        self.v = v
        self.residual_capacity = residual_capacity
        self.reverse = None
        self.residual_edge = None  

    def update_residual_capacity(self,flow):
        """ 
        Function description:
        This function updates the residual capacity edge and reverse edge in the residual graph.

        Approach description:
        - Decrement the residual capacity of the edge by the flow amount.
        - Check if the current edge has a corresponding reverse edge. If so, increment the reverse edge's residual capacity by the flow amount.

        Input:
        - flow (int): The amount of flow to be sent through the edge.

        Precondition:
        - The flow must be a non-negative integer.

        Postcondition:
        - The residual capacity of the edge is updated.
        - The reverse edge's residual capacity is updated if it exists.     

        Time complexity:
        - O(1) as it only involves updating the residual capacity of the edge and its reverse edge.

        Auxillary Space complexity:
        - O(1) as it only uses a fixed amount of extra space regardless of the input size.
        """

        self.residual_capacity -= flow 
        if self.reverse:
            self.reverse.residual_capacity += flow

        
def assign(preferences, places):
    """ 
    Function description: 
    The assign function solves the problem of assigning participants to activities during a weekend getaway, ensuring that each activity has leaders 
    with experience and other participants based on their preferences. The function constructs a network flow graph and uses the Ford-Fulkerson algorithm 
    to find a maximum flow, which corresponds to the optimal assignment of participants to activities.

    Approach description:
    - Graph Construction:
        - Build a flow network where each participant is represented by a node, and each activity is represented by two nodes: one for experienced participants and one for additional participants.
        - A source node is connected to all participant nodes, and the activity nodes are connected to a sink node.
        - Edges are added from participants to activities based on preferences: participants with preference 2 (experienced) have edges to both the experienced and additional participant nodes of the activity, 
          while participants with preference 1 have edges only to the additional participant node.
        - Edges from the experienced activity nodes to the sink have a capacity of 2, while edges from the additional participant nodes to the sink have a capacity of places[j] - 2.

    - Network Flow Construction:
        - The Ford-Fulkerson algorithm is used to calculate the maximum flow in the network, which represents the maximum number of participants that can be assigned to activities while respecting the capacity constraints (activity sizes).

    - Assignment Construction:
        - After finding the maximum flow, the assignments are constructed by tracing the flow from the source to the sink.
        - If the maximum flow equals the total number of places available in activities, a valid assignment exists
        - It then traces the flow through the network to determine the specific assignment of each participant to an activity.
        - If a valid assignment is found, the function returns a list where each entry corresponds to an activity, and the participants assigned to that activity are listed. If no valid assignment is found, the function returns None. 

    Input:
    - preferences (list): A list of lists of integers representing the preferences of participants for activities.
    - places (list): A list of integers representing the number of places available for each activity,including both experienced and additional participants.

    Precondition:
    - preferences is a list of lists of integers with a length equal to the number of participants.
    - places is a list of integers with a length equal to the number of activities.
    - Each value in preferences is either 0, 1, or 2.
    - Each value in places is a positive integer greater than or equal to 2 (since each activity requires at least two participants).

    Postcondition:
    - If a valid assignment is found, the function returns a list where each entry corresponds to an activity, and the participants assigned to that activity are listed. 
    - If no valid assignment is found, the function returns None. 

    Return:
    - assignments (list): A list of lists of integers representing the assignments of participants to activities.

    Time complexity:
    - Graph Construction:
        - Number of vertices:
            - There are n participants and m activities. Each activity has 2 vertices, one for experienced participants and one for additional participants.
            - There are also 2 additional vertices, source and sink.
            - The total number of vertices will be N + 2m + 2.
            - Since we can assume the number of activities will be at most n/2.
            - Therefore the vertices will be just, V = N

        - Number of edges:
            - Each participant is connected to the source (N edges)
            - Each participant is connected to one or two activity vertices based on their preferences,leading to almost total of 2N edges, 
              since each participant can have a preference for one or two activities.
            - Each activity vertex (experienced and super) is connected to the sink (2m edges)
            - Therefore, the total number of edges, E = N + 2N + 2M 
                                                     = 3N + 2M.
            - Since we can assume the number of activities will be at most N/2.
            - Therefore, total number of edges can be E=  3N
    
    - Ford-Fulkerson Algorithm:
        - The Ford Fulkerson algorithm runs in O(N^2*M) time, which also can be interpreted as O(V*E^2)
        - Since V = N and E= 3N
        - Therefore the total time complexity will be O(N*N^2), which is O(N^3)

    Auxillary Space Complexity:
    - Initialise the Network Flow graph takes O(N+M) space
    - Running Ford Fulkerson takes O(N+M) space
    - Assigning participants takes up O(N) space in the worst case, where all participants are assigned
    - Since number of activities is at most N/2
    - Therefore the total auxillary space will be O(N)
        
        
    """
    num_participants = len(preferences)
    num_activities = len(places)
    
    # Calculate the size of the flow network based on activities
    network_size = num_participants + 2 * num_activities + 2
    flow_network = Network_Flow(network_size)
    
    source = network_size - 2
    sink = network_size - 1
    preference_count = [0] * num_participants
    for i in range(num_participants ):
        for j in range(num_activities):
            if preferences[i][j] == 2:
                preference_count[i] += 1
    
    #Add edges from source to each participant with capacity 1
    for i in range(num_participants):
        flow_network.add_edge(source, i, 1)
        
    
   #Add edges from participants to activities based on preferences
    for i in range(num_participants):
        for j in range(num_activities):
            if preferences[i][j] == 2:  # Experienced
                flow_network.add_edge(i, num_participants + j, 1)
                if places[j] > 2:   
                    flow_network.add_edge(i, num_participants + num_activities + j, 1)
            elif preferences[i][j] == 1:  # Interested but not experienced
                flow_network.add_edge(i, num_participants + num_activities + j, 1)
    
    # Add edges from activity vertices (experienced and additional) to sink
    for j in range(num_activities):
        flow_network.add_edge(num_participants + j, sink, 2) 
        flow_network.add_edge(num_participants + num_activities + j, sink, places[j] - 2)
    
    max_flow = flow_network.ford_fulkerson(source, sink,preference_count)
    
    #Check if max_flow equals total participants needed
    if max_flow == sum(places):
        # Construct the assignments if a valid assignment is found
        assignments = [[] for _ in range(num_activities)]
        for i in range(num_participants):
            assigned_activity = None
            for edge in flow_network.vertices[i].edges:
                if edge.flow > 0 and edge.v != source:
                    activity = edge.v - num_participants
                    if activity < num_activities:  # preference 2-only vertex
                        assigned_activity = activity  # Prioritize experienced participants
                    elif assigned_activity is None:  # If not experienced, check super vertex
                        assigned_activity = activity - num_activities 
            if assigned_activity is not None:
                assignments[assigned_activity].append(i)
        return assignments
        

    return None  # No valid assignment found


""" 
Task 2: Spell Checker 
"""

class SpellChecker:
    def __init__(self, filename):
        """ 
        Function description:
        - Initializes the SpellChecker object using a file containing words and constructs a Trie for efficient word lookup.

        Approach description:
        - Initialize the words list to store words from the file.
        - Initialize the SpellChecker object with the given filename.
        - Call the process_file method to read the file and extract words.
        - Construct a Trie with the extracted words for efficient word lookup.

        Input:
        - filename: The name of the txt file containing words to be used for spell checking.

        Preconditions:
        - The filename must be a txt file.

        Postconditions:
        - The words list will contain all the words from the file.
        - The Trie will be constructed with the words from the file for efficient lookup.

        Time complexity: 
        -O(T) where T is the total number of characters in the file.
        -O(T) for processing the file and extracting words.
        -O(T) for inserting each word into the Trie.
        -Total: O(T)

        Auxillary space complexity:
        -O(T) for storing the words list.
        -O(T) for storing the Trie nodes.
        -Total: O(T)

        """
        self.words = []
        self.process_file(filename) 
        self.trie = Trie(self.words)
        for key in self.words:
            self.trie.insert(key)

    def process_file(self,filename):
        """ 
        Function description:
        - Processes the file to extract words and store them in the words list.

        Approach description:
        - Open the file and read line by line.
        - Use regular expression to split each line into words
        - Non-alphanumeric characters are used as delimiters for splitting.
        - Add the extracted words to the words list.
        - Remove any empty strings from the words list.
        
        Input:
        - filename: The name of the txt file containing words to be used for spell checking.

        Preconditions:
        - The filename must be a txt file.

        Postconditions:
        - The words list will contain all the words from the file.

        Time complexity:
        - O(T) where T is the total number of characters in the file.
        - O(T) for reading the file and splitting each line into words.
        - O(T) for removing empty strings from the words list.
        - Total: O(T)

        Auxillary space complexity:
        - O(T) for storing the words list.
        """
        filename = open(filename,"r")
        for line in filename:
            self.words.extend(re.split(r'[^a-zA-Z0-9]+',line.strip()))
        filename.close()
        self.words = [word for word in self.words if word]
        

    def check(self,word):
        """ 
        Function description:
        - Checks the Trie for the best matching words for the input word.

        Approach description:
        - Call the recursive_search method to search for the word in the Trie to find potential matches for the input word
        - It then processes the result to compile a list of the 3 best matching words.
        - If the word is not found, return an empty list.
        - If the word is found, return the best 3 words.

        Input:
        - word(str): The word to be checked for in the Trie.
        
        Returns:
        - best_words(list): The best 3 matching words for the input word.

        Preconditions:
        - The word must be a valid alphanumeric string.

        Postconditions:
        - The best 3 matching words for the input word are returned.

        Time complexity:
        - O(M+U) for the recursive_search method, where M is the length of the input word and U is the number of best words in the Trie.
        - O(1) for checking the length of the word_info array and iterating through it to get the best words. 
          It can be considered as O(1) since the word_info array is fixed at 3, regardless of the input word length.
        - Total: O(M+U),where M is the length of the input word and U is the number of best words in the Trie.
        - 


        Auxillary space complexity:
        - O(U) for array returned by the recursive_search method, where U is the number of best words in the Trie.

        
        """

        word_info = self.trie.recursive_search(word)
        best_words = []
        if len(word_info) == 0:
            return []
        for i in range(3):
            if word_info[i] is not None:
                word_index = word_info[i][0]
                actual_word = self.trie.words[word_index]
                best_words.append(actual_word)

        return best_words


class Trie:
    def __init__(self,words,size = 63):
        """ 
        Function description:
        - Initializes the Trie object to store words efficiently for lookup

        Approach description:
        - Initialize the root node.
        - Store the words list for reference.
        - set the size of link array for each node to 63(including numbers,lowercase and uppercase letters)

        Input:
        - words: List of words to be stored in the Trie.
        - size: Size of the character set (default is 63).

        Precondition:
        - The words list must contain all the words to be stored in the Trie.

        Postconditions:
        - The root node is initialized.
        - The words list is stored for reference.
        - The size of the link array for each node is set to 63.

        Time complexity:
        - O(1) to initialize the root node.
        - O(1) to store the words list.
        - O(1) to set the size of the link array for each node.
        - Total: O(1)

        Auxillary space complexity:
        - O(1) to store the words list.
        - O(1) to store the root node.
        - Total: O(1)
        """
        self.root = Node()
        self.words = words

    def char_to_index(self,char):
        """ 
        Function description:
        - Converts a character to its corresponding index in the link array.
        - designed to map characters to specific indices in a link array of size 63 in the Trie node, 
          enabling efficient storage and retrieval of different types of characters.


        Approach description:
        - If character is a digit, the function subtracts the ASCII value of '0'(48) from the ASCII value of the character and adds 1 to get the index. 1 is added because index cannot be 0 as 0 is used for terminal node.
        - If character is a uppercase letter, the function subtracts the ASCII value of 'A'(65) from the ASCII value of the character and adds 11 to get the index. 11 is added because there are 10 digits before uppercase letters.
        - If character is a lowercase letter, the function subtracts the ASCII value of 'a'(97) from the ASCII value of the character and adds 37 to get the index. 37 is added because there are 10 digits and 26 uppercase letters before lowercase letters.
        - Special character, Dollar sign($) is mapped to 0. This serves as a terminal node, which is used to denote the end of a word
        - Digits(0-9) are mapped to 1-10.
        - Uppercase letters(A-Z) are mapped to 11-36.
        - Lowercase letters(a-z) are mapped to 37-62.


        Input:
        - char: The character to be converted to its corresponding index.

        Preconditions:
        - char must be a valid alphanumeric character or $

        Postconditions:
        - The index of the character in the link array is returned.

        Time complexity:
        - O(1) to convert the character to its corresponding index.
        
        Auxillary space complexity:
        - O(1) to store the index.
        
        """
        if char == "$":
            return 0
        elif char.isdigit():
            return ord(char) - 48 + 1
        elif char.isupper():
            return ord(char) - 65 + 11
        elif char.islower():
            return ord(char) - 97 + 37
        else:
            raise ValueError("Character not supported")
        
        
    def recursive_search(self,word):
        """ 
        Function description:
        - Initiate a recursive search through the Trie to find the best 3 words that match the input word.
        - Make use of the recursive function _search_recursive to search for the word.

        Input:
        - word: The word to be searched for in the Trie.

        Preconditions:
        - The word must be a valid alphanumeric string.

        Postconditions:
        - The best 3 words that match the input word are returned.

        Time complexity:
        - O(M) for each recursive call for each character, where M is the length of the input word.
        - At each node, the method combines the best words from current.best_words and returned_best_words using the _combine_best_words method. 
          Since each of these lists has a maximum of 3 entries, the combination operation takes O(1) time (constant time) because it operates on a fixed number of entries.
        - Total: O(M),where M is the length of the input word.

        Auxillary space complexity:
        - O(M) for the recursive call stack, where M is the length of the input word.
        """
        return self._search_recursive(self.root,word,0)
    def _search_recursive(self,current,word,height):
        """ 
        Function description:
        - A helper function used by recursive_search to recursively search for the word in the Trie.
        - Navigates through the Trie using the characters of the word as indices.

        Approach description:
        - Base case:
            - If the current height (character position) is equal to the length of the word, this indicates that we have processed all characters of the word.
            - If the word already exists(no terminal link), return the best_words array from the current node.
            - If the word does not exist(terminal link), return an empty array.
        - Recursive case:
            - If the search has not reached the end of the word, it retrieves the current character (char = word[height]).
            - Converts the current character to its corresponding index using the char_to_index method.
            - If the corresponding link (child node) does not exist, return an empty array.
            - Move to the corresponding child node(next character node) and recursively call _search_recursive with the updated height.
            - Combine the best_words array from the current node and the returned best_words array from the recursive search.
            - Returns the combined best_words array from the current and next level of trie 
        
        Input:
        - current: The current node in the Trie.
        - word: The word to be searched for.
        - height: The current height (character position) in the word.

        Preconditions:
        - The word must be a valid alphanumeric string.
        - The height must be a non-negative integer.

        Postconditions:
        - The best_words array from the current and next level of trie is combined.
        
        Return:
            - If the returned best_words array is empty, return an empty array.
            - If the returned best_words array is not empty, return the combined best_words array.

        Time complexity:
        - O(M) for each recursive call for each character, where M is the length of the input word.
        - O(1) for the combine_best_words function, as it operates on fixed-size arrays.(up to 3 words)
        - Total: O(M), where M is the length of the input word.

        Auxillary space complexity:
        - O(M) for the recursive call stack, where M is the length of the input word.
        """
        if height == len(word):
            if current.link[0] is not None:
                return []
            else:
                return current.best_words
        char = word[height]
        index = self.char_to_index(char)
        if current.link[index] is None: #this word does not exist
            return current.best_words
        next_node = current.link[index]
        returned_best_words =  self._search_recursive(next_node,word,height+1)
        combined_best_words = self._combine_best_words(current.best_words,returned_best_words)
        #if exact same word is found
        if len(returned_best_words) == 0:
            return returned_best_words
        return combined_best_words


    def _combine_best_words(self,current_best_words,returned_best_words):
        """ 
        Function description:
        - Combines the best_words array from the current node and the returned best_words array from the recursive search.
        - Ensures that the combined best_words array contains unique words and has a maximum of 3 entries.
        - Stops once the combined_best_words array is filled with 3 unique words. 

        Approach description:
        - Initialize combined_best_words to hold up to 3 entries.
        - Use pos to track the index for inserting unique tuples.
        - Helper function is_unique checks if the word index already exists in combined_best_words.
        - Iterate through returned_best_words and add unique words to combined_best_words.
        - Iterate through current_best_words and add unique words to combined_best_words.

        Input:
        - current_best_words: The best_words array from the current node.
        - returned_best_words: The best_words array from the recursive search.

        Preconditions:
        - The current_best_words and returned_best_words arrays must contain tuples with word index and frequency.

        Postconditions:
        - The combined_best_words array will contain up to 3 unique tuples merged from current_best_words and returned_best_words.

        Time complexity:
        - O(3) to iterate through the returned_best_words array to check for unique words, it's bounded by a constant, since the array size is fixed at 3.
        - O(3) per call to is_unique when this function is called, it iterates through the returned_best_words array to add unique words.
        - Processing all entries in returned_best_words array takes O(3*3) = O(9) time.
        - Iterates over current_best_words (up to 3 entries) and performs the same uniqueness check using is_unique. Processing all entries in current_best_words array takes O(3*3) = O(9) time.
        - O(9+9) = O(18)
        - O(18) is bounded by a constant, since the array size is fixed at 3.
        - Total: O(1)
        
        Auxillary space complexity:
        - O(1) to store the combined_best_words array.
        - No new data structures proportional to the input size are created during the processing of entries from current_best_words or returned_best_words. 
        - The method reuses the input data and stores results in the fixed-size combined_best_words list.
        - Total: O(1)
        """
        combined_best_words = [None] * 3  # Initialize to hold up to 3 entries
        pos = 0  # Position to track the index for inserting unique tuples
        for word in returned_best_words:
            if word is not None and self.is_unique(word, combined_best_words):
                combined_best_words[pos] = word
                pos += 1
                if pos >= 3: 
                    return combined_best_words

        # Then add unique words from current_best_words
        for word in current_best_words:
            if word is not None and self.is_unique(word, combined_best_words):
                combined_best_words[pos] = word
                pos += 1
                if pos >= 3:  
                    return combined_best_words

        return combined_best_words  
        
            
    def is_unique(self,word, combined):
        """ 
        Function description:
        This is a helper function to check if the word index already exists in combined_best_words

        Approach description:
        - Iterate through combined_best_words and check if the word index already exists.
        - If it exists, return False.
        - If it does not exist, return True.

        Input:
        - word: The word index to be checked.
        - combined: The combined_best_words array.

        Time complexity:
        - O(3) to iterate through the combined_best_words array to check for unique words, it's bounded by a constant, since the array size is fixed at 3.
        
        Auxillary space complexity:
        - O(1) to store the result.
        """
        for existing_word in combined:
            if existing_word is not None and existing_word[0] == word[0]:
                return False
        return True

    
    def insert(self,key):
        """ 
        Function description:
        - Inserts a word into the Trie.

        Approach description:
        - Start at the root node.
        - Recursively insert each character of the word into the Trie,creating new nodes as necessary.
        - Make use of the recursive function _insert_recursive to insert the word.

        Input:
        - key: The word to be inserted into the Trie.

        Preconditions:
        - The word must be a valid alphanumeric string.

        Postconditions:
        - The word is inserted into the Trie, with its frequency and word index updated.

        Time complexity:
        - The _insert_recursive call takes O(M) time where M is the length of word and k is due to calling update_best_word function
        - So the time complexity will be O(M)
        
        Auxillary space complexity:
        - O(M) due to recursive node creation
        """

        current = self.root
        self._insert_recursive(current,key,0)

    def _insert_recursive(self,current,key,height):
        """ 
        Function description:
        - A helper function used by insert to recursively insert a word into the Trie, starting from the root node and at a specific height

        Approach description:
        - Base case:
            - If the current height (character position) is equal to the length of the word, this indicates that we have processed all characters of the word.
            - If the word already exists, update its frequency.
            - If the word does not exist, create a new terminal node and insert the word.
        - Recursive case:
            - Convert the current character to its corresponding index.
            - If the corresponding link (child node) does not exist, create a new node.
            - Move to the corresponding child node(next character node) and recursively call _insert_recursive with the updated height.
        - Return:
            - As the recursion unwinds, the word information (word index and frequency) for the current node is passed back up the chain of recursive calls.
            - This information is used to update the best_words array at each node.
        - Update the best_words array at each node, ensuring that it holds the top 3 words which satisfy the best_words constraints

        Input:
        - current: The current node in the Trie.
        - key: The word to be inserted.
        - height: The current height (character position) in the word.

        Preconditions:
        - The word must be a valid alphanumeric string.

        Postconditions:
        - The word is inserted into the Trie, with its frequency and word index updated.
        - The best_words array at each node is updated to hold the top 3 words which satisfy the best_words constraints.

        Time complexity:
        - The recursive call takes O(M) time where M is the length of the word
        - At each level of recursion, the update_best_words function is called, which contributes to O(1) time
        - Total Complexity: O(M)

        Auxillary space complexity:
        - O(M) due to recursive node creation.

        """
        if height == len(key):
            #Move through the terminal node
            if current.link[0] is not None: # if the word already exists(terminal exists)
                current = current.link[0]
            else:  
                current.link[0] = Node() 
                current = current.link[0]
            current.word_index = self.words.index(key)
            current.frequency += 1 
            current.word_info = (current.word_index,current.frequency)
            return current.word_info # ensure word index in available to all nodes 
        
        char = key[height]

        index = self.char_to_index(char)
        current.char_order = index 
        
        if current.link[index] is None: 
            current.link[index] = Node() 

        current = current.link[index]
        current.height += 1
        current.word_info = self._insert_recursive(current,key,height+1)
        self._update_best_words(current,current.word_info)
        return current.word_info 

    def _update_best_words(self,node,current_word_info):
        """ 
        Function description:
        - Updates the best_words array at each node to hold the top 3 words which satisfy the best_words constraints

        Approach description:
        - Check for empty slots in the best_words array. If a slot is available, insert the new word entry.
        - If the list is full, compare the new word entry with the existing words in the best_words array.
        - The is_better function is used to determine if the new word entry is better than the current word at the appropriate position.
        - If the new word entry has a higher frequency or is lexicographically smaller, replace the existing word at the appropriate position.
        - If the new word entry is not better than any existing word, do nothing.
        - The best_words array is updated in place, with the most relevant words at the top.

        Input:
        - node: The current node in the Trie.
        - current_word_info: The word information for the current word.

        Preconditions:
        - The best_words array is initialized with None values.
        - The node exists and current_word_info is a tuple containing the word index and frequency of the current word.

        Postconditions:
        - The best_words array is updated with the new word entry based on the best_words constraints. i) more shared prefix, ii) higher frequency, iii) shorter length

        Time complexity:
        - O(1) for inserting or updating the best_words array as the best words array is limited to 3 words, so it's bounded by a constant time.
        - O(1) for comparing the new word entry with the existing words in the best_words array.
        - O(1) for calling the _is_better function, since the best words array is bounded by a size of 3, so I only need to perform comparisons up to 3 words.
          The time spent on updating best words at each node does not depend on the total number of words or the structure of the Trie.
        - 0(1) for shifting elements if the new word entry is better than the current best entry, sice the list has a fixed size of 3.
        - Total: O(1)

        Auxillary space complexity:
        - O(1) to store the new_entry, current_word_info, and the best_words array.

        """
        # Assuming word_index and word information is available
        new_entry = current_word_info
        for i in range(3):
            if node.best_words[i] is None:
                node.best_words[i] = new_entry
                return 
            if node.best_words[i][0] == new_entry[0]:  # Same word
                if new_entry[1] > node.best_words[i][1]: 
                    node.best_words[i] = new_entry  
                return 
        
            # Check if the new entry is better than the current one
            if self._is_better(node, new_entry, node.best_words[i]):
                # Shift from the last index to the current index
                for j in range(2, i, -1):
                    node.best_words[j] = node.best_words[j - 1]
                node.best_words[i] = new_entry 
                return 

    def _is_better(self, node, new_word_entry, current_best_entry):
        """ 
        Function description:
        - Compares two words based on frequency and lexicographical order to determine if the new word entry is better than the current best entry

        Approach description:
        - Compare by frequency:
            - If the new word has a higher frequency, it is better.
            - If the frequencies are the same, compare the lexicographical order of the words.
        - Compare by lexicographical order:
            - If the frequencies are the same, compare the lexicographical order of the words.
            - Iterates through each character of both words until it finds a difference and rank them accordingly
            - If the shorter word runs out of characters, the comparison should treat the shorter word as being padded with null characters (\0, ASCII 0) at the end.
            - The comparison should continue until the end of the longer word.

        - If the words are identical (even after padding), return False (neither is "better")

        Input:
        - node: The current node in the Trie.
        - new_word_entry: The word information for the new word entry.
        - current_best_entry: The word information for the current best entry.

        Preconditions:
        - The new_word_entry and current_best_entry are tuples containing the word index and frequency of the new word and current best word respectively.

        Postconditions:
        - Return True if the new_word_entry is better than the current_best_entry based on the best_words constraints.

        Time complexity:
        - O(1) to compare the frequencies of two words.
        - O(k) to compare the lexicographical order of the words, where M is the maximum length of the two words.

        Auxillary space complexity:
        - O(1) to store the result.
        
        """
        # Extract word information
        new_word_index, new_frequency = new_word_entry
        current_word_index, current_frequency = current_best_entry
        new_word = self.words[new_word_index]
        current_word = self.words[current_word_index]

        #  Compare by frequency (higher is better)
        if new_frequency > current_frequency:
            return True
        elif new_frequency < current_frequency:
            return False

        #Compare by word length (shorter is better)
        max_len = max(len(new_word), len(current_word))

        for i in range(max_len):
            if i < len(new_word):
                new_char = new_word[i] 
            else:
                new_char = '\0'
            if i < len(current_word):
                current_char = current_word[i]
            else:
                current_char = '\0'
            new_char_index = ord(new_char)
            current_char_index = ord(current_char)
            if new_char_index < current_char_index:
                return True
            elif new_char_index > current_char_index:
                return False

    # If the words are identical (even after padding), return False
        return False

class Node:
    def __init__(self):
        """ 
        Function description:
        - Initializes the Node object with default values for the Trie.

        Approach description:
        - Initialize the link array to None with a size of 63.
        - Initialize the height to 0.
        - Initialize the frequency to 0.
        - Initialize the char_order to None.
        - Initialize the word_index to None.
        - Initialize the best_words array to None with a size of 3.
        - Initialize the word_info to None.

        Time complexity:
        - O(1) to initialize the link array.
        - O(1) to initialize the height.
        - O(1) to initialize the frequency.

        Auxillary space complexity:
        - O(1) to store the link array.
        - O(1) to store the height.
        - O(1) to store the frequency.
        - O(1) to store the char_order.
        - O(1) to store the word_index.
        - O(1) to store the best_words array.
        - O(1) to store the word_info.
        """
        self.link = [None]*63
        self.height = 0 
        self.frequency = 0
        self.char_order = None
        self.word_index = None
        self.best_words = [None]*3
        self.word_info = None



