#include <iostream>
#include <random>
#include <tuple>
#include <list>
#include <cstring>
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
    delete connections;
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
    char *name;
    float fitness;
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
    normal_distribution<float> neg_norm_distribution;
    normal_distribution<float> pos_norm_distribution;

public:
    Genome(char *name);
    ~Genome();
    Genome(Genome *other);
    void InitGenome(int numInputs, int numOutputs);
    void Load(const char *filename);
    void Save(const char *filename);
    void Mutate();
    void Crossover(Genome *other);
    int FeedForward(float *input_image);

    void SetName(char *name) { this->name = name; }
    char *GetName() { return name; }
    void SetFitness(float fitness) { this->fitness = fitness; }
    float GetFitness() { return fitness; }

    // TODO:!
    Node *FindRandomNodeWithConnection();
};

Genome::Genome(char *name)
{
    this->name = new char[strlen(name) + 1];
    strcpy(this->name, name);
    this->numInputs = 0;
    this->numOutputs = 0;
    this->numHidden = 0;
    neg_norm_distribution = normal_distribution<float>(-1.0, 1.0);
    pos_norm_distribution = normal_distribution<float>(0.0, 1.0);
}

Genome::~Genome()
{
    cout << "Genome destroyed!";
}

Genome::Genome(Genome *other)
{
    this->name = new char[strlen(other->name) + 1];
    strcpy(this->name, other->name);
    this->fitness = 0;
    this->numNodes = other->numNodes;
    this->numInputs = other->numInputs;
    this->numOutputs = other->numOutputs;
    this->numHidden = other->numHidden;
    // TODO: need to copy nodes, bias, inputs, outputs, and hidden but preserver the connections between the new nodes
    // this->nodes = other->nodes;
    // this->bias = other->bias;
    // this->inputs = other->inputs;
    // this->outputs = other->outputs;
    // this->hidden = other->hidden;
    neg_norm_distribution = normal_distribution<float>(-1.0, 1.0);
    pos_norm_distribution = normal_distribution<float>(0.0, 1.0);
}

void Genome::InitGenome(int numInputs, int numOutputs)
{
    // cout << "Initializing genome with:" << numInputs << " inputs and " << numOutputs << " outputs." << endl;
    this->numInputs = numInputs;
    this->numOutputs = numOutputs;
    // cout << "set numInputs: " << this->numInputs << endl;
    // cout << "set numOutputs: " << this->numOutputs << endl;
    numNodes = numInputs + numOutputs + 1;
    // cout << "numNodes: " << numNodes << endl;
    nodes = new Node *[numNodes];
    // cout << "created nodes array" << endl;
    bias = new Node(0, 1);
    // cout << "created bias node" << endl;
    nodes[0] = bias;
    // cout << "put bias node in node list as 0th node" << endl;
    for (int i = 0; i < numNodes - 1; i++)
    {
        nodes[i + 1] = new Node(i, -INFINITY);
    }
    // cout << "initialized the rest if the nodes" << endl;
    inputs = new Node *[numInputs];
    for (int i = 0; i < numInputs; i++)
    {
        inputs[i] = nodes[i + 1];
    }
    // cout << "set inputs" << endl;
    outputs = new Node *[numOutputs];
    for (int i = 0; i < numOutputs; i++)
    {
        outputs[i] = nodes[numInputs + i + 1];
    }
    for (int i = 0; i < numOutputs; i++)
    {
        outputs[i]->AddConnection(bias, neg_norm_distribution(generator));
    }
    // cout << "Genome initialized!" << endl;
}

void Genome::Load(const char *filename)
{
    numInputs = 3;
    numHidden = 2;
    numOutputs = 3;
    numNodes = numInputs + numHidden + numOutputs;
    Node *node0 = new Node(1, -INFINITY);
    Node *node1 = new Node(2, -INFINITY);
    Node *node2 = new Node(3, -INFINITY);
    Node *node3 = new Node(4, -INFINITY);
    Node *node4 = new Node(5, -INFINITY);
    Node *node5 = new Node(6, -INFINITY);
    Node *node6 = new Node(7, -INFINITY);
    Node *node7 = new Node(8, -INFINITY);

    nodes = new Node *[8]{node0, node1, node2, node3, node4, node5, node6, node7};
    inputs = new Node *[3]{node0, node1, node2};
    hidden = new Node *[2]{node3, node4};
    outputs = new Node *[3]{node5, node6, node7};

    // Connections
    node3->AddConnection(node0, 0.5);
    node3->AddConnection(node1, 0.25);

    node4->AddConnection(node3, 2.0);
    node4->AddConnection(node2, 1.0);

    node5->AddConnection(node3, 1.0);

    node6->AddConnection(node3, 0.5);

    node7->AddConnection(node4, 0.1);

    cout << "Genome loaded!" << endl;
}

void Genome::Save(const char *filename)
{
    cout << "Genome saved!" << endl;
}

Node *Genome::FindRandomNodeWithConnection()
{
    // cout << "Finding random node with connection" << endl;
    int random_node_index = 0;
    Node *random_node = nodes[random_node_index];
    while (random_node->GetConnections()->empty())
    {
        random_node_index = rand() % numNodes;
        random_node = nodes[random_node_index];
    }
    return random_node;
}

void Genome::Mutate()
{
    cout << "Genome mutated!" << endl;

    float mutation = pos_norm_distribution(generator);
    if (mutation < 0.5)
    {
        // mutation 1: adjust the weight of a connection
        Node *random_node = FindRandomNodeWithConnection();
        list<tuple<Node *, float, bool>> *connections = random_node->GetConnections();
        int random_connection_index = rand() % connections->size();
        // tuple<Node *, float, bool> random_connection = connections->  
    }
    else if (mutation < 0.6)
    {
        // mutation 2: add a new connection
    }
    else if (mutation < 0.7)
    {
        // mutation 3: add a new node
    }
    else if (mutation < 0.8)
    {
        // mutation 4: disable a connection
    }
    else if (mutation < 0.9)
    {
        // mutation 5: enable a connection
    }
    else
    {
        // mutation 6: change an activation function
    }

    // if mutation < 0.5:
    //     # mutation 1: adjust the weight of a connection
    //     # print("Mutating weight")
    //     new_genome.connections[random.randint(
    //         0, len(new_genome.connections)-1)].weight += random.uniform(-0.1, 0.1)
    // elif mutation < 0.6:
    //     # mutation 2: add a new connection
    //     # print("Mutating connection")
    //     # TODO: check that the connection does not already exist
    //     # TODO: check that the connection does not create a cycle
    //     # TODO: check that the connection does not connect to an input node
    //     raise NotImplementedError
    //     pass
    // elif mutation < 0.7:
    //     # mutation 3: add a new node
    //     # adding a node will disable a connection and create two new connections
    //     # print("Mutating node")
    //     raise NotImplementedError
    //     pass
    // elif mutation < 0.8:
    //     # mutation 4: disable a connection
    //     # print("Mutating disable")
    //     # TODO: check that the connection is not already disabled
    //     # TODO: check that the connection is not the only connection to an output node
    //     raise NotImplementedError
    //     pass
    // elif mutation < 0.9:
    //     # mutation 5: enable a connection
    //     # print("Mutating enable")
    //     # TODO: check that the connection is not already enabled
    //     # TODO: check that enabling the connection does not create a cycle
    //     raise NotImplementedError
    //     pass
    // else:
    //     # mutation 6: change an activation function
    //     # print("Mutating activation")
    //     # TODO: check that the node is not an input or output node
    //     # TODO: check that the node is not the bias node
    //     # TODO: check that the node does not already use the activation function
    //     raise NotImplementedError
    //     pass
}

void Genome::Crossover(Genome *other)
{
    cout << "Genome crossed over!" << endl;
}

int Genome::FeedForward(float *input_image)
{
    /**
     * 'Feed forward' algorithm implemented as a recursive search starting from the output nodes.
     */
    // cout << "Initial state:" << endl;
    // for (int i = 0; i < numNodes; i++)
    // {
    //     if (i < 10 || i > numNodes - 10)
    //     {
    //         cout << i << ":" << nodes[i]->GetValue() << endl;
    //     }
    // }

    // Set input values
    for (int i = 0; i < numInputs; i++)
    {
        this->inputs[i]->SetValue(input_image[i]);
    }

    // cout << "State after setting input values:" << endl;
    // for (int i = 0; i < numNodes; i++)
    // {
    //     if (i < 10 || i > numNodes - 10)
    //     {
    //         cout << i << ":" << nodes[i]->GetValue() << endl;
    //     }
    // }

    // Run feed forward
    for (int i = 0; i < numOutputs; i++)
    {
        outputs[i]->CalculateValue();
    }

    // cout << "State after feed forward:" << endl;
    // for (int i = 0; i < numNodes; i++)
    // {
    //     if (i < 10 || i > numNodes - 10)
    //     {
    //         cout << i << ":" << nodes[i]->GetValue() << endl;
    //     }
    // }

    int max_index = 0;
    float max_value = -INFINITY;
    for (int i = 0; i < numOutputs; i++)
    {
        if (outputs[i]->GetValue() > max_value)
        {
            max_value = outputs[i]->GetValue();
            max_index = i;
        }
    }
    // cout << "Max index: " << max_index << " Max value: " << max_value << endl;
    return max_index;
}

// void Genome::LoadTestCase()
// {
//     // print initial state
//     cout << "Initial state:" << endl;
//     for (int i = 0; i < numNodes; i++)
//     {
//         cout << i << ":" << nodes[i]->GetValue() << endl;
//     }
//     // feed forward
//     this->FeedForward(new float[3]{0, 1, 0.25});
//     // print state after feed forward
//     cout << "State after feed forward:" << endl;
//     for (int i = 0; i < numNodes; i++)
//     {
//         cout << i << ":" << nodes[i]->GetValue() << endl;
//     }
// }

extern "C"
{
    Genome *NewGenome(char *name)
    {
        return new Genome(name);
    }
    void DeleteGenome(Genome *genome)
    {
        delete genome;
    }
    Genome *CopyGenome(Genome *genome)
    {
        return genome;
    }
    void InitGenome(Genome *genome, int numInputs, int numOutputs)
    {
        genome->InitGenome(numInputs, numOutputs);
    }
    void LoadGenome(Genome *genome, char *filename)
    {
        genome->Load(filename);
    }
    void SaveGenome(Genome *genome, const char *filename)
    {
        genome->Save(filename);
    }
    void MutateGenome(Genome *genome)
    {
        genome->Mutate();
    }
    void CrossoverGenome(Genome *genome, Genome *other)
    {
        genome->Crossover(other);
    }
    int FeedForwardGenome(Genome *genome, float *input_image)
    {
        return genome->FeedForward(input_image);
    }

    void SetName(Genome *genome, char *name)
    {
        genome->SetName(name);
    }
    char *GetName(Genome *genome)
    {
        return genome->GetName();
    }

    void SetFitness(Genome *genome, float fitness)
    {
        genome->SetFitness(fitness);
    }
    float GetFitness(Genome *genome)
    {
        return genome->GetFitness();
    }
}

// int main()
// {
//     // genome->InitGenome();
//     // genome->FeedForward(new float[3]{1, 2, 3});

//     // EXAMPLE 1

//     // vars
//     int numInputs = 3;
//     int numHidden = 2;
//     int numOutputs = 3;
//     int numNodes = numInputs + numHidden + numOutputs;
//     Genome *genome = new Genome(numInputs, numHidden, numOutputs);

//     // Nodes
//     Node *node0 = new Node(1, 0);
//     Node *node1 = new Node(2, 1);
//     Node *node2 = new Node(3, 0.25);
//     Node *node3 = new Node(4, -INFINITY);
//     Node *node4 = new Node(5, -INFINITY);
//     Node *node5 = new Node(6, -INFINITY);
//     Node *node6 = new Node(7, -INFINITY);
//     Node *node7 = new Node(8, -INFINITY);

//     // Node lists
//     Node **nodes = new Node *[8]{node0, node1, node2, node3, node4, node5, node6, node7};
//     Node **inputs = new Node *[3]{node0, node1, node2};
//     Node **hidden = new Node *[2]{node3, node4};
//     Node **outputs = new Node *[3]{node5, node6, node7};

//     // Connections
//     node3->AddConnection(node0, 0.5);
//     node3->AddConnection(node1, 0.25);

//     node4->AddConnection(node3, 2.0);
//     node4->AddConnection(node2, 1.0);

//     node5->AddConnection(node3, 1.0);

//     node6->AddConnection(node3, 0.5);

//     node7->AddConnection(node4, 0.1);

//     // Load config
//     // genome->LoadConfig(numNodes, nodes, numInputs, inputs, numOutputs, outputs, numHidden, hidden);

//     // print state before feed forward
//     cout << "Initial state:" << endl;
//     for (int i = 0; i < numNodes; i++)
//     {
//         cout << i << ":" << nodes[i]->GetValue() << endl;
//     }

//     // feed forward
//     genome->FeedForward(new float[3]{0, 1, 0.25});

//     // print state after feed forward
//     cout << "State after feed forward:" << endl;
//     for (int i = 0; i < numNodes; i++)
//     {
//         cout << i << ":" << nodes[i]->GetValue() << endl;
//     }
//     return 0;
// }