#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <tuple>
#include <list>
#include <cstring>
#include <cstdlib>
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
    bool connections_disabled = false;

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
    void Save(string *filename);
    void Load(string *filename);
    void LoadConnections(string *filename, Node **nodes, int numNodes);
    list<tuple<Node *, float, bool>> *GetConnections() { return connections; };
};

Node *GetNodeWithId(Node **nodes, int numNodes, int id)
{
    for (int i = 0; i < numNodes; i++)
    {
        if (nodes[i]->GetId() == id)
        {
            return nodes[i];
        }
    }
    return NULL;
}

Node::Node(int id, float value)
{
    this->id = id;
    this->value = value;
    ready = false;
    connections = new list<tuple<Node *, float, bool>>();
    connections_disabled = false;
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

void Node::Save(string *filename)
{
    ofstream MyFile(*filename);
    MyFile << id << endl;
    MyFile << value << endl;
    MyFile << connections->size() << endl;
    for (tuple<Node *, float, bool> conn : *connections)
    {
        MyFile << get<0>(conn)->GetId() << " " << get<1>(conn) << " " << get<2>(conn) << endl;
    }
    MyFile.close();
}

void Node::Load(string *filename)
{
    /*
    Initial part of the load process, the second function LoadConnections will be called after this function
    */
    ifstream MyFile(*filename);
    MyFile >> id;
    MyFile >> value;
    MyFile.close();
}

void Node::LoadConnections(string *filename, Node **nodes, int numNodes)
{
    ifstream MyFile(*filename);
    MyFile >> id;
    MyFile >> value;
    int num_connections;
    MyFile >> num_connections;
    for (int i = 0; i < num_connections; i++)
    {
        int from_id;
        float weight;
        bool enabled;
        MyFile >> from_id >> weight >> enabled;
        Node *from_node = GetNodeWithId(nodes, numNodes, from_id);
        connections->push_back(make_tuple(from_node, weight, enabled));
    }
    MyFile.close();
}

class Genome
{
private:
    string *name;
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
    Genome(string *name);
    ~Genome();
    Genome(Genome *other);
    void InitGenome(int numInputs, int numOutputs);
    void Load(string *filename);
    void Save(string *filename);
    void Mutate();
    void Crossover(Genome *other);
    int FeedForward(float *input_image);

    void SetName(string *name) { this->name = name; }
    string *GetName() { return name; }
    void SetFitness(float fitness) { this->fitness = fitness; }
    float GetFitness() { return fitness; }
    void PrintInfo();

    // TODO:!
    Node *FindRandomNodeWithEnabledConnection();
};

Genome::Genome(string *name)
{
    this->name = new string(name->c_str());
    this->numInputs = 0;
    this->numOutputs = 0;
    this->numHidden = 0;
    neg_norm_distribution = normal_distribution<float>(-1.0, 1.0);
    pos_norm_distribution = normal_distribution<float>(0.0, 1.0);
}

Genome::~Genome()
{
    // delete[] name;
    // delete[] nodes;
    // delete[] inputs;
    // delete[] outputs;
    // delete[] hidden;
}

Genome::Genome(Genome *other)
{
    this->name = new string(other->name->c_str());
    this->fitness = 0;
    this->numNodes = other->numNodes;
    if (this->numNodes == 0)
    {
        this->numNodes = 1000;
    }
    this->numInputs = other->numInputs;
    this->numOutputs = other->numOutputs;
    this->numHidden = other->numHidden;

    // TODO: need to copy nodes, bias, inputs, outputs, and hidden but preserver the connections between the new nodes
    this->nodes = new Node *[numNodes];
    for (int i = 0; i < numNodes; i++)
    {
        this->nodes[i] = new Node(other->nodes[i]->GetId(), other->nodes[i]->GetValue());
    }
    this->bias = this->nodes[0];
    this->inputs = new Node *[numInputs];
    for (int i = 0; i < numInputs; i++)
    {
        int id = other->inputs[i]->GetId();
        for (int j = 0; j < numNodes; j++)
        {
            if (nodes[j]->GetId() == id)
            {
                this->inputs[i] = nodes[j];
                break;
            }
        }
    }
    this->outputs = new Node *[numOutputs];
    for (int i = 0; i < numOutputs; i++)
    {
        int id = other->outputs[i]->GetId();
        for (int j = 0; j < numNodes; j++)
        {
            if (nodes[j]->GetId() == id)
            {
                this->outputs[i] = nodes[j];
                break;
            }
        }
    }
    this->hidden = new Node *[numHidden];
    for (int i = 0; i < numHidden; i++)
    {
        int id = other->hidden[i]->GetId();
        for (int j = 0; j < numNodes; j++)
        {
            if (nodes[j]->GetId() == id)
            {
                this->hidden[i] = nodes[j];
                break;
            }
        }
    }

    // Copy connections
    // for every node in the other genome, find the corresponding node in this genome and copy the connections
    for (int i = 0; i < numNodes; i++)
    {
        int current_id = nodes[i]->GetId();
        Node *current_node = GetNodeWithId(other->nodes, numNodes, current_id);
        // list<tuple<Node *, float, bool>> *connections = current_node->GetConnections();
        list<tuple<Node *, float, bool>> *connections = current_node->connections;
        for (tuple<Node *, float, bool> conn : *connections)
        {
            int from_id = get<0>(conn)->GetId();
            Node *from_node = GetNodeWithId(other->nodes, numNodes, from_id);
            float weight = get<1>(conn);
            bool enabled = get<2>(conn);
            this->nodes[i]->AddConnection(from_node, weight);
        }
    }

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

void Genome::Load(string *dir)
{

    ifstream MyFile(*dir + "/genome.txt");

    cout << "Loading genome from: " << *dir << "/genome.txt" << endl;

    // first line: numInputs, numHidden, numOutputs
    MyFile >> numInputs >> numHidden >> numOutputs;

    // Note: node[0] is always the bias node
    // second line: input node ids
    int *input_ids = new int[numInputs];
    for (int i = 0; i < numInputs; i++)
    {
        MyFile >> input_ids[i];
    }

    // third line: hidden node ids
    int *hidden_ids = new int[numHidden];
    for (int i = 0; i < numHidden; i++)
    {
        MyFile >> hidden_ids[i];
    }

    // fourth line: output node ids
    int *output_ids = new int[numOutputs];
    for (int i = 0; i < numOutputs; i++)
    {
        MyFile >> output_ids[i];
    }

    // for each node: load node
    nodes = new Node *[numNodes];
    nodes[0] = new Node(0, 1);
    bias = nodes[0];
    for (int i = 0; i < numNodes - 1; i++)
    {
        string *node_filename = new string(*dir + "/node" + to_string(i) + ".txt");
        nodes[i + 1] = new Node(i, -INFINITY);
        nodes[i + 1]->Load(node_filename);
    }

    // for each node: load connections
    for (int i = 0; i < numNodes - 1; i++)
    {
        string *node_filename = new string(*dir + "/node" + to_string(i) + ".txt");
        nodes[i + 1]->LoadConnections(node_filename, nodes, numNodes);
    }

    // set inputs, outputs, and hidden

    inputs = new Node *[numInputs];

    for (int i = 0; i < numInputs; i++)
    {
        inputs[i] = GetNodeWithId(nodes, numNodes, input_ids[i]);
    }

    outputs = new Node *[numOutputs];

    for (int i = 0; i < numOutputs; i++)
    {
        outputs[i] = GetNodeWithId(nodes, numNodes, output_ids[i]);
    }

    hidden = new Node *[numHidden];

    for (int i = 0; i < numHidden; i++)
    {
        hidden[i] = GetNodeWithId(nodes, numNodes, hidden_ids[i]);
    }

    cout << "Genome loaded!" << endl;
}

void Genome::Save(string *dir)
{
    // Create file
    ofstream MyFile(*dir + "/genome.txt");

    cout << "Saving genome to: " << *dir << "/genome.txt" << endl;

    // first line: numInputs, numHidden, numOutputs
    MyFile << numInputs << " " << numHidden << " " << numOutputs << endl;

    // Note: node[0] is always the bias node
    // second line: input node ids
    for (int i = 0; i < numInputs; i++)
    {
        MyFile << inputs[i]->GetId() << " ";
    }
    MyFile << endl;

    // third line: hidden node ids
    for (int i = 0; i < numHidden; i++)
    {
        MyFile << hidden[i]->GetId() << " ";
    }
    MyFile << endl;

    // fourth line: output node ids
    for (int i = 0; i < numOutputs; i++)
    {
        MyFile << outputs[i]->GetId() << " ";
    }
    MyFile << endl;

    MyFile.close();

    // for each node: save node
    for (int i = 0; i < numNodes; i++)
    {
        string *node_filename = new string(*dir + "/node" + to_string(i) + ".txt");
        nodes[i]->Save(node_filename);
        free(node_filename);
    }
    cout << "Genome saved!" << endl;
}

Node *Genome::FindRandomNodeWithEnabledConnection()
{
    if (numNodes == 0)
    {
        cout << "No nodes in genome, cannot FindRandomNodeWithEnabledConnection" << endl;
        return NULL;
    }
    int i = 0;
    // possible infinite loop, need to make sure there is always at least one enabled connection in the genome
    while (true)
    {
        i++;
        if (i > 100)
        {
            cout << "No enabled connections in genome, cannot mutate" << endl;
            return NULL;
        }
        // int random_node_index = (int)(pos_norm_distribution(generator) * numNodes);
        int random_node_index;
        // Node *random_node = nodes[random_node_index];
        Node *random_node;
        // while (random_node->GetConnections()->empty() || random_node->connections_disabled)
        while (true)
        {
            // cout << "numNodes: " << numNodes << endl;
            // random_node_index = (int)(pos_norm_distribution(generator) * numNodes);
            random_node_index = rand() % numNodes;
            random_node = nodes[random_node_index];
            // list<tuple<Node *, float, bool>> *conns = random_node->GetConnections();
            // cout << random_node->id << endl;
            list<tuple<Node *, float, bool>> *conns = random_node->connections;
            // cout << "node: " << random_node->GetId() << "num_connections" << conns->size() << endl;
            if (conns->size() > 0)
            {
                break;
            }
        }
        // check if all connections are disabled
        bool all_connections_disabled = true;
        // for (tuple<Node *, float, bool> conn : *random_node->GetConnections())
        for (tuple<Node *, float, bool> conn : *random_node->connections)
        {
            if (get<2>(conn))
            {
                all_connections_disabled = false;
                break;
            }
        }
        if (all_connections_disabled)
        {
            random_node->connections_disabled = true;
            continue;
        }
        return random_node;
    }
}

void Genome::Mutate()
{
    // cout << "Genome mutating" << endl;

    // float mutation = pos_norm_distribution(generator);
    // if (mutation < 0.5)
    // {
    // mutation 1: adjust the weight of a connection
    Node *random_node = FindRandomNodeWithEnabledConnection();
    if (random_node == NULL)
    {
        cout << "No enabled connections in genome, cannot mutate" << endl;
        return;
    }
    list<tuple<Node *, float, bool>> *connections = random_node->GetConnections();
    int random_connection_index = rand() % connections->size();
    std::list<tuple<Node *, float, bool>>::iterator it = connections->begin();
    advance(it, random_connection_index);
    tuple<Node *, float, bool> random_connection = *it;
    connections->erase(it);
    Node *from_node = get<0>(random_connection);
    float weight = get<1>(random_connection);
    bool enabled = get<2>(random_connection);
    weight += pos_norm_distribution(generator);
    connections->push_back(make_tuple(from_node, weight, enabled));
    // }
    // else if (mutation < 0.6)
    // {
    //     // mutation 2: add a new connection
    // }
    // else if (mutation < 0.7)
    // {
    //     // mutation 3: add a new node
    // }
    // else if (mutation < 0.8)
    // {
    //     // mutation 4: disable a connection
    // }
    // else if (mutation < 0.9)
    // {
    //     // mutation 5: enable a connection
    // }
    // else
    // {
    //     // mutation 6: change an activation function
    // }

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
    // cout << "Genome mutated!" << endl;
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

    // Set input values
    for (int i = 0; i < numInputs; i++)
    {
        this->inputs[i]->SetValue(input_image[i]);
    }

    // Run feed forward
    for (int i = 0; i < numOutputs; i++)
    {
        outputs[i]->CalculateValue();
    }

    // Find the output node with the highest value
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

    return max_index;
}

void Genome::PrintInfo()
{
    cout << "Genome name: " << name << endl;
    cout << "Genome fitness: " << fitness << endl;
    cout << "Genome numNodes: " << numNodes << endl;
    cout << "Genome numInputs: " << numInputs << endl;
    cout << "Genome numOutputs: " << numOutputs << endl;
    cout << "Genome numHidden: " << numHidden << endl;
}

extern "C"
{
    Genome *NewGenome(char *name)
    {
        return new Genome(new string(name));
    }
    void DeleteGenome(Genome *genome)
    {
        delete genome;
    }
    Genome *CopyGenome(Genome *genome)
    {
        return new Genome(genome);
    }
    void InitGenome(Genome *genome, int numInputs, int numOutputs)
    {
        genome->InitGenome(numInputs, numOutputs);
    }
    void LoadGenome(Genome *genome, char *filename)
    {
        genome->Load(new string(filename));
    }
    void SaveGenome(Genome *genome, char *filename)
    {
        genome->Save(new string(filename));
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
    void PrintGenomeInfo(Genome *genome)
    {
        genome->PrintInfo();
    }
    void SetName(Genome *genome, char *name)
    {
        genome->SetName(new string(name));
    }
    const char *GetName(Genome *genome)
    {
        return genome->GetName()->c_str();
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
