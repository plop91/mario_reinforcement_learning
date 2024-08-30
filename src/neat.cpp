#include <iostream>
#include <random>
#include <tuple>
#include <list>
using namespace std;

class Node
{
    int id;
    float value;
    // float bias;
    // float *weights;
    // int *connections;
    list<tuple<Node *, float>> *connections; // tuple<from_node, weight>
    bool ready;

public:
    Node(int id, float value);
    ~Node();
    void CalculateValue();
    void AddConnection(Node *node, float weight);
    void SetValue(float value)
    {
        this->value = value;
        this->ready = true;
    }
    void IsReady() { ready = true; }
    void Reset();
    tuple<Node *, float> GetConnection(int index) { return connections->front(); };
    list<tuple<Node *, float>> *GetConnections() { return connections; };
};

Node::Node(int id, float value)
{
    cout << "Node created!" << endl;
    this->id = id;
    this->value = value;
    ready = false;
    connections = new list<tuple<Node *, float>>();
}

Node::~Node()
{
    cout << "Node destroyed!";
}

void Node::AddConnection(Node *node, float weight)
{
    connections->push_back(make_tuple(node, weight));
}

void Node::CalculateValue()
{
    if (connections->empty())
    {
        ready = true;
        return;
    }
    value = 0;
    for (tuple<Node *, float> conn : *connections)
    {
        if (!get<0>(conn)->ready)
        {
            get<0>(conn)->CalculateValue();
        }
        value += get<0>(conn)->value * get<1>(conn);
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

    //DEBUG
    // void addNode(Node *node) { nodes[0] = node; }
    void AddBias(Node *node) { bias = node; }
    void AddConnection(Node *from, Node *to, float weight) { to->AddConnection(from, weight); }
};

Genome::Genome(int numInputs, int numOutputs, int numHidden)
{
    cout << "Genome created!";
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

class Neat
{
public:
    Neat();
    ~Neat();
};

int main()
{
    Genome *genome = new Genome(3,2,3);
    genome->InitGenome();
    genome->FeedForward(new float[3]{1, 2, 3});
    return 0;
}