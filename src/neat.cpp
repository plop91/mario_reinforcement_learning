#include <iostream>
#include <random>
#include <tuple>
#include <list>
using namespace std;

/*
Examples of configuration settings from:
https://github.com/CodeReclaimers/neat-python/blob/master/examples/circuits/config

[NEAT]
fitness_criterion     = max     # unknown
fitness_threshold     = -0.01   # unknown
pop_size              = 500     # population size
reset_on_extinction   = False   # if True, reset all species if all species are extinct

[CircuitGenome]
# component type options
component_default      = resistor           # unknown
component_mutate_rate  = 0.1                # unknown
component_options      = resistor diode     # unknown

# component value options
value_init_mean          = 4.5      # unknown
value_init_stdev         = 0.5      # unknown
value_max_value          = 6.0      # unknown
value_min_value          = 3.0      # unknown
value_mutate_power       = 0.1      # unknown
value_mutate_rate        = 0.8      # unknown
value_replace_rate       = 0.1      # unknown

# genome compatibility options
compatibility_disjoint_coefficient = 1.0    # unknown
compatibility_weight_coefficient   = 1.0    # unknown

# connection add/remove rates
conn_add_prob           = 0.2   # probability of adding a connection
conn_delete_prob        = 0.2   # probability of deleting a connection

# connection enable options
enabled_default         = True  # probability of a connection being enabled/disabled
enabled_mutate_rate     = 0.02  # probability of a connection being enabled/disabled

# node add/remove rates
node_add_prob           = 0.1   # probability of adding a node
node_delete_prob        = 0.1   # probability of deleting a node

# network parameters
num_inputs              = 3  # number of input nodes
num_outputs             = 1 # number of output nodes

[DefaultSpeciesSet]
compatibility_threshold = 2.0   # some threshold for compatibility between species, not sure if I'll use this

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 15

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
*/

class Node
{
    // private:
public:
    int id;
    float value;
    // float bias;
    // float *weights;
    // int *connections;
    list<tuple<Node *, float, bool>> *connections; // tuple<from_node, weight>
    bool ready;

// public:
    Node(int id, float value);
    ~Node();
    void CalculateValue();
    void AddConnection(Node *node, float weight);
    void SetValue(float value)
    {
        this->value = value;
        this->ready = true;
    }
    int GetId() { return id; }
    float GetValue() { return value; }
    void IsReady() { ready = true; }
    void Reset();
    tuple<Node *, float, bool> GetConnection(int index) { return connections->front(); };
    list<tuple<Node *, float, bool>> *GetConnections() { return connections; };
};

Node::Node(int id, float value)
{
    this->id = id;
    this->value = value;
    ready = false;
    connections = new list<tuple<Node *, float, bool>>();
}

Node::~Node()
{
    cout << "Node destroyed!";
}

void Node::AddConnection(Node *node, float weight)
{
    connections->push_back(make_tuple(node, weight, true));
}

void Node::CalculateValue()
{
    // Check if node has any incoming connections, if not, set ready to true and do not update value
    if (connections->empty())
    {
        ready = true;
        return;
    }

    // Check if all incoming connections are disabled, if so, set ready to true and do not update value
    bool all_connections_disabled = true;
    for (tuple<Node *, float, bool> conn : *connections)
    {
        if (get<2>(conn))
        {
            all_connections_disabled = false;
            break;
        }
    }

    // If all incoming connections are disabled, set ready to true and do not update value
    if (all_connections_disabled)
    {
        ready = true;
        return;
    }

    // Calculate value recursively
    value = 0;
    for (tuple<Node *, float, bool> conn : *connections)
    {
        if (get<2>(conn))
        {
            if (!get<0>(conn)->ready)
            {
                get<0>(conn)->CalculateValue();
            }
            value += get<0>(conn)->value * get<1>(conn);
        }
    }
    ready = true;
}

void Node::Reset()
{
    ready = false;
}

class Genome
{
private:
    int numNodes;
    int numInputs;
    int numOutputs;
    int numHidden;
    Node **nodes;
    Node *bias;
    Node **inputs;
    Node **outputs;
    Node **hidden;

    default_random_engine generator;
    normal_distribution<float> distribution;

public:
    Genome(int, int, int);
    ~Genome();
    void InitGenome();
    void FeedForward(float *input_image);
    void Mutate();

    // DEBUG
    // void AddNode(Node *node) { nodes[0] = node; }
    // void AddBias(Node *node) { bias = node; }
    // void AddConnection(Node *from, Node *to, float weight) { to->AddConnection(from, weight); }
    void LoadConfig(int numNodes, Node **nodes, int numInputs, Node **inputs, int numOutputs, Node **outputs, int numHidden, Node **hidden)
    {
        this->numNodes = numNodes;
        this->numInputs = numInputs;
        this->numOutputs = numOutputs;
        this->numHidden = numHidden;
        this->nodes = nodes;
        this->inputs = inputs;
        this->outputs = outputs;
        this->hidden = hidden;
    }
    Node **GetOutputNodes() { return outputs; }
};

Genome::Genome(int numInputs, int numOutputs, int numHidden)
{
    distribution = normal_distribution<float>(-1.0, 1.0);
}

Genome::~Genome()
{
    cout << "Genome destroyed!";
}

void Genome::InitGenome()
{
    cout << "Genome initialized!";
    int totalNodes = numInputs + numOutputs + numHidden + 1;
    nodes = new Node *[totalNodes];
    bias = new Node(0, 1);
    nodes[0] = bias;
    for (int i = 0; i < totalNodes - 1; i++)
    {
        nodes[i + 1] = new Node(i, -INFINITY);
    }
    for (int i = 0; i < numInputs; i++)
    {
        inputs[i] = nodes[i + 1];
    }
    for (int i = 0; i < numOutputs; i++)
    {
        outputs[i] = nodes[numInputs + numHidden + i + 1];
    }
    for (int i = 0; i < numOutputs; i++)
    {
        bias->AddConnection(outputs[i], distribution(generator));
    }
}

void Genome::FeedForward(float *input_image)
{
    for (int i = 0; i < numInputs; i++)
    {
        inputs[i]->SetValue(input_image[i]);
    }
    for (int i = 0; i < numOutputs; i++)
    {
        outputs[i]->CalculateValue();
    }
}

extern "C"
{
    Node *NewNode(int id, float value)
    {
        return new Node(id, value);
    }
    int NodeGetId(Node *node)
    {
        return node->GetId();
        // return 0;
    }
    float NodeGetValue(Node *node)
    {
        // return node->GetValue();
        return 0;
    }
    void NodeAddConnection(Node *node, Node *to, float weight)
    {
        node->AddConnection(to, weight);
    }
    Genome *NewGenome(int numInputs, int numOutputs, int numHidden)
    {
        return new Genome(numInputs, numOutputs, numHidden);
    }
}

int main()
{
    // genome->InitGenome();
    // genome->FeedForward(new float[3]{1, 2, 3});

    // EXAMPLE 1

    // vars
    int numInputs = 3;
    int numHidden = 2;
    int numOutputs = 3;
    int numNodes = numInputs + numHidden + numOutputs;
    Genome *genome = new Genome(numInputs, numHidden, numOutputs);

    // Nodes
    Node *node0 = new Node(1, 0);
    Node *node1 = new Node(2, 1);
    Node *node2 = new Node(3, 0.25);
    Node *node3 = new Node(4, -INFINITY);
    Node *node4 = new Node(5, -INFINITY);
    Node *node5 = new Node(6, -INFINITY);
    Node *node6 = new Node(7, -INFINITY);
    Node *node7 = new Node(8, -INFINITY);

    // Node lists
    Node **nodes = new Node *[8]{node0, node1, node2, node3, node4, node5, node6, node7};
    Node **inputs = new Node *[3]{node0, node1, node2};
    Node **hidden = new Node *[2]{node3, node4};
    Node **outputs = new Node *[3]{node5, node6, node7};

    // Connections
    node3->AddConnection(node0, 0.5);
    node3->AddConnection(node1, 0.25);

    node4->AddConnection(node3, 2.0);
    node4->AddConnection(node2, 1.0);

    node5->AddConnection(node3, 1.0);

    node6->AddConnection(node3, 0.5);

    node7->AddConnection(node4, 0.1);

    // Load config
    genome->LoadConfig(numNodes, nodes, numInputs, inputs, numOutputs, outputs, numHidden, hidden);

    // print state before feed forward
    cout << "Initial state:" << endl;
    for (int i = 0; i < numNodes; i++)
    {
        cout << i << ":" << nodes[i]->GetValue() << endl;
    }

    // feed forward
    genome->FeedForward(new float[3]{0, 1, 0.25});

    // print state after feed forward
    cout << "State after feed forward:" << endl;
    for (int i = 0; i < numNodes; i++)
    {
        cout << i << ":" << nodes[i]->GetValue() << endl;
    }
    return 0;
}