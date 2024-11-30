#I wrote this one a long time ago. AI has helped me clean it up! 
#make sure stack is imported 
from stack import Stack
from queue import Queue
import heapq


class Graph:
    class Vertex:
        def __init__(self, label):
            self.label = label
            self.neighbors = []  # List of tuples: (neighbor_label, weight)

    def __init__(self):
        self.vertice_list = []  # List of Vertex objects
        self.vertex_labels = []  # List of vertex labels
        self.size = 0
        self.start = None

    def is_empty(self):
        return self.size == 0

    def add_vertex(self, label):
        if not isinstance(label, str):
            raise ValueError("Vertex label must be a string")
        if label in self.vertex_labels:
            raise ValueError(f"Vertex '{label}' already exists")

        new_vertex = self.Vertex(label)
        self.vertice_list.append(new_vertex)
        self.vertex_labels.append(label)
        if self.size == 0:
            self.start = new_vertex
        self.size += 1

    def add_edge(self, src, dest, weight):
        if not isinstance(weight, (float, int)):
            raise ValueError("Weight must be a number")
        if src not in self.vertex_labels or dest not in self.vertex_labels:
            raise ValueError("Source or destination vertex not found")

        src_vertex = next(v for v in self.vertice_list if v.label == src)
        dest_vertex = next(v for v in self.vertice_list if v.label == dest)
        src_vertex.neighbors.append((dest, float(weight)))

    def get_weight(self, src, dest):
        if src not in self.vertex_labels or dest not in self.vertex_labels:
            raise ValueError("Source or destination vertex not found")

        src_vertex = next(v for v in self.vertice_list if v.label == src)
        for neighbor, weight in src_vertex.neighbors:
            if neighbor == dest:
                return weight

        return float("inf")  # No path exists

    def dfs(self, starting_vertex):
        if starting_vertex not in self.vertex_labels:
            raise ValueError("Starting vertex not found in the graph")

        visit = {v.label: False for v in self.vertice_list}
        stack = Stack()
        stack.push(starting_vertex)
        result = []

        while not stack.is_empty():
            current = stack.pop()
            if not visit[current]:
                visit[current] = True
                result.append(current)
                current_vertex = next(v for v in self.vertice_list if v.label == current)
                neighbors = sorted([n[0] for n in current_vertex.neighbors], reverse=True)
                for neighbor in neighbors:
                    if not visit[neighbor]:
                        stack.push(neighbor)

        for vertex in result:
            yield vertex

    def bfs(self, starting_vertex):
        if starting_vertex not in self.vertex_labels:
            raise ValueError("Starting vertex not found in the graph")

        visit = {v.label: False for v in self.vertice_list}
        queue = Queue()
        queue.put(starting_vertex)
        result = []

        while not queue.empty():
            current = queue.get()
            if not visit[current]:
                visit[current] = True
                result.append(current)
                current_vertex = next(v for v in self.vertice_list if v.label == current)
                neighbors = sorted([n[0] for n in current_vertex.neighbors])
                for neighbor in neighbors:
                    if not visit[neighbor]:
                        queue.put(neighbor)

        for vertex in result:
            yield vertex

    def dijkstra_shortest_path(self, start_label):
        """Dijkstra's algorithm for finding the shortest path in a graph."""
        if start_label not in self.vertex_labels:
            raise ValueError(f"Vertex '{start_label}' not found in the graph")

        # Initialize the shortest path tree and distances
        distances = {vertex.label: float('inf') for vertex in self.vertice_list}
        distances[start_label] = 0
        previous_vertices = {vertex.label: None for vertex in self.vertice_list}
        pq = [(0, start_label)]  # Priority queue with (distance, vertex)

        while pq:
            current_distance, current_vertex_label = heapq.heappop(pq)

            # Early termination if current_distance is larger than known shortest path
            if current_distance > distances[current_vertex_label]:
                continue

            current_vertex = next(v for v in self.vertice_list if v.label == current_vertex_label)
            for neighbor_label, weight in current_vertex.neighbors:
                distance = current_distance + weight

                # If shorter path found, update the distance and the previous vertex
                if distance < distances[neighbor_label]:
                    distances[neighbor_label] = distance
                    previous_vertices[neighbor_label] = current_vertex_label
                    heapq.heappush(pq, (distance, neighbor_label))

        # Return the shortest paths
        return distances, previous_vertices

    def __str__(self):
        if self.is_empty():
            return "Graph is empty"

        output = ["digraph G {"]
        for vertex in self.vertice_list:
            for neighbor, weight in sorted(vertex.neighbors, key=lambda x: x[0]):
                output.append(f'    {vertex.label} -> {neighbor} [label="{weight}", weight="{weight}"];')
        output.append("}")
        return "\n".join(output)


def main():
    G = Graph()

    # Add vertices
    for label in ["A", "B", "C", "D", "E", "F"]:
        G.add_vertex(label)

    # Add edges
    G.add_edge("A", "F", 9)
    G.add_edge("A", "B", 2)
    G.add_edge("B", "C", 8)
    G.add_edge("B", "D", 15)
    G.add_edge("B", "F", 6)
    G.add_edge("F", "B", 6)
    G.add_edge("C", "D", 1)
    G.add_edge("E", "C", 7)
    G.add_edge("E", "D", 3)
    G.add_edge("F", "E", 3)

    # Print graph structure
    print(G)
    print()

    # Test DFS
    print("Starting DFS from vertex A:")
    for vertex in G.dfs("A"):
        print(vertex, end=" ")
    print("\n")

    # Test BFS
    print("Starting BFS from vertex A:")
    for vertex in G.bfs("A"):
        print(vertex, end=" ")
    print("\n")

    # Test Dijkstra's Shortest Path
    print("Starting Dijkstra's from vertex A:")
    distances, previous = G.dijkstra_shortest_path("A")
    print("Distances:", distances)
    print("Previous vertices:", previous)


if __name__ == "__main__":
    main()
