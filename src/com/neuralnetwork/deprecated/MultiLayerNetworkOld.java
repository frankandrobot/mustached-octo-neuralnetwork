package com.neuralnetwork.deprecated;

import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.interfaces.OldINeuralNetwork;
import com.neuralnetwork.core.neuron.NVector;
import com.neuralnetwork.core.neuron.Neuron;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class MultiLayerNetworkOld
{
    protected LayorInfo[] aLayers;
    protected int numberLayers;

    protected ExampleInfo[] aExamples;
    protected int numberExamples;

    /**
     * Aka ... error
     */
    protected NVector vTotalDifferenceSquared;

    final protected double alpha; /** momentum **/
    final protected double eta; /** learning parameter **/
    final protected int numberIterations;

    protected IActivationFunction.IDifferentiableFunction phi;

    public MultiLayerNetworkOld(Builder builder)
    {
        this.eta = builder.eta;
        this.alpha = builder.alpha;
        this.phi = builder.phi;
        this.numberIterations = builder.numberIterations == 0 ? 1000 : builder.numberIterations;

        initializeLayers(builder);
    }

    protected LayorInfo getLayer(int i)
    {
        return aLayers[i];
    }

    static public class Builder
    {
        private IActivationFunction.IDifferentiableFunction phi;
        protected OldINeuralNetwork[] aLayers;
        protected Double alpha;
        protected Double eta;
        protected int numberIterations;

        public Builder setGlobalActivationFunction(IActivationFunction.IDifferentiableFunction phi)
        {
            this.phi = phi;
            return this;
        }

        public Builder setLayers(OldINeuralNetwork... aLayers)
        {
            this.aLayers = aLayers;
            return this;
        }

        public Builder setMomentumParam(Double alpha)
        {
            this.alpha = alpha;
            return this;
        }

        public Builder setLearningParam(Double eta)
        {
            this.eta = eta;
            return this;
        }

        public Builder setIterations(int numberIterations)
        {
            this.numberIterations = numberIterations;
            return this;
        }
    }

    /**
     * (vExampleInput, vExpected) form the training example
     */
    protected class ExampleInfo
    {
        protected NVector vExampleInput;
        protected NVector vExpected;

        /**
         * Used to store the actual output for the given example input
         */
        protected NVector vActual;
        /**
         * Stores (vActual - vExpected).(vActual - vExpected) for the given example
         * If vActual = od and vExpected = td,
         * then these are vectors (od = [od_x,od_y], td = [td_x, td_y])
         * so then (od - td).(od - td) = (od_x - td_x)^2 + (od_y - td_y)^2
         * Hence the name #differenceSquared
         */
        protected double differenceSquared;
    }

    /**
     * This class wraps the SingleLayorNeuralNetwork
     * with additional information.
     *
     * Basically it works per layer
     * and you get back info about its individual neurons
     */
    protected class LayorInfo
    {
        final public OldINeuralNetwork<NVector,NVector,Neuron> layer;
        /**
         * For each neuron k in the layer,
         * store v_k (induced local field).
         */
        public NVector vInducedLocalField;
        /**
         * For each neuron k in the layer,
         * store value of impulse function.
         */
        public NVector vImpulseFunction;
        /**
         * Store the inputs (y_k) from the previous layer.
         */
        public NVector vInput;
        /**
         * For each neuron k in the layer,
         * aPrevWeights[k] is its previous set of weights.
         * aPrevWeights[k].get(j) is neuron k's jth previous weight.
         */
        public NVector[] aPrevWeights;

        /**
         * For each neuron k in the layer,
         * store its gradient
         */
        public NVector vGradients;

        /**
         * For neuron k's jth weight in the layer,
         * aWeightAdjustments[k].get(j) is the quantity to add to its jth weight
         * when adjusting the weight with the backprop algorithm
         */
        public NVector[] aWeightAdjustments;

        public LayorInfo(OldINeuralNetwork layer)
        {
            this.layer = layer;
            this.vInducedLocalField = new NVector().setSize(this.layer.getNumberOfNeurons());
            this.vImpulseFunction = new NVector().setSize(this.layer.getNumberOfNeurons());
            this.vGradients = new NVector().setSize(this.layer.getNumberOfNeurons());
            this.aPrevWeights = new NVector[this.layer.getNumberOfNeurons()];
            this.aWeightAdjustments = new NVector[this.layer.getNumberOfNeurons()];
            int len=0;
            for(Neuron neuron:this.layer)
            {
                this.aPrevWeights[len] = new NVector().setSize(neuron.getNumberOfWeights());
                this.aWeightAdjustments[len] = new NVector().setSize(neuron.getNumberOfWeights());
                ++len;
            }
        }
    }

    protected class DebugOutput
    {
        String weights()
        {
            String rslt = "";
            for(int neuron=0; neuron< aLayers[0].layer.getNumberOfNeurons(); neuron++)
            {
                rslt += String.format("%20s | %20s %n",
                        getNeuron(aLayers[0].layer, neuron),
                        getNeuron(aLayers[1] != null ? aLayers[1].layer : null, neuron));
            }
            return rslt;
        }

        private String getNeuron(OldINeuralNetwork network,
                                 int neuron)
        {
            if (network != null)
                return neuron < network.getNumberOfNeurons()
                        ? network.getNeuron(neuron).toString()
                        : "";
            return "";
        }

        /**
         * Dumps the debug info for the given backprop iteration
         *
         * @param iteration given iteration of the backprop algorithm
         */
        protected void backpropDump(int iteration)
        {
            System.out.println("====================================================================");
            System.out.println("Iteration = "+ iteration);
            System.out.format("Weights =%n%s", weights());
            System.out.format("Error = %s %n", vTotalDifferenceSquared.sumOfCoords());
            System.out.format("%5s %20s %20s%n", "i", "Expected", "Actual");
            for(int i=0; i< numberExamples; ++i)
                System.out.format("%5s %20s %20s%n",
                        i,
                        aExamples[i].vExpected,
                        aExamples[i].vActual);
        }
    }

    protected void initializeLayers(Builder builder)
    {
        numberLayers = builder.aLayers.length;

        aLayers = new LayorInfo[numberLayers];

        for(int i=0; i<numberLayers; i++)
            aLayers[i] = new LayorInfo(builder.aLayers[i]);
    }

    public void setupExampleInfo(NVector... aInputExpected)
    {
        if (aInputExpected.length % 2 != 0)
            throw new IllegalArgumentException();

        numberExamples = aInputExpected.length / 2;

        aExamples = new ExampleInfo[numberExamples];

        //save aInputExpected
        for(int i=0; i< numberExamples; i++)
        {
            aExamples[i] = new ExampleInfo();
            aExamples[i].vExampleInput = aInputExpected[2*i];
            aExamples[i].vExpected = aInputExpected[2*i+1];
        }

        //setup error
        vTotalDifferenceSquared = new NVector().setSize(numberExamples);
    }

    public void backpropagation(double errorBound, NVector... aInputExpected)
    {
        if (aInputExpected.length % 2 != 0)
            throw new IllegalArgumentException();

        //initialization
        setupExampleInfo(aInputExpected);

        DebugOutput debugOutput = new DebugOutput();
        int iteration = 1;

        while( backpropagation() > errorBound && iteration <= numberIterations)
        {
            debugOutput.backpropDump(iteration);
            iteration++;
        }
        debugOutput.backpropDump(iteration);
    }

    public NVector output(NVector input)
    {
        if (aExamples == null)
        {
            setupExampleInfo(input, new NVector(0));
        }

        return output(0, 0, input);
    }

    /**
     * Trains the network using pairs of inputs/expected values
     *
     * @return sum of the total differnce squared (error)
     */
    protected double backpropagation()
    {
        resetWeightAdjustments();

        for(int i=0; i< numberExamples; i++)
        {
            forwardPropagation(i);
            constructGradients(i);
            saveWeightAdjustments();
        }

        adjustWeights();

        for(int i=0; i< numberExamples; i++)
        {
            constructErrorFunction(i);
        }

        return vTotalDifferenceSquared.sumOfCoords();
    }

    protected void constructErrorFunction(int example)
    {
        NVector vActual = forwardPropagation(example);
        NVector vExpected = aExamples[example].vExpected;
        aExamples[example].differenceSquared = vExpected.subtract(vActual).dotProduct();
        vTotalDifferenceSquared.set(example, aExamples[example].differenceSquared);
    }

    protected NVector forwardPropagation(int example)
    {
        return output(example, 0, aExamples[example].vExampleInput);
    }

    /**
     * Find the actual output for the given example.
     *
     * Side effect: store stuff (induced local field, impulse function, etc) in the layer info
     *
     *
     * @param example
     * @param layer the example's layer
     * @return output
     */
    protected NVector output(int example, int layer, NVector input)
    {
        LayorInfo layorInfo = aLayers[layer];

        //save info for back propagation
        layorInfo.vInput = new NVector(input, 1f); //tack on bias at end

        constructInducedLocalField(layorInfo);
        constructImpulseFunction(layorInfo);

        if (layer < aLayers.length - 1)
        {
            return output(example, layer+1, aLayers[layer].vImpulseFunction);
        }
        else
        {
            aExamples[example].vActual = new NVector(aLayers[layer].vImpulseFunction);
        }
        return layorInfo.vImpulseFunction;
    }

    private void constructInducedLocalField(LayorInfo layorInfo)
    {
        //calculate v_k's
        int len=0;
        for(Neuron neuron:layorInfo.layer)
        {
            layorInfo.vInducedLocalField.set(len++, neuron.rawoutput(layorInfo.vInput));
        }
    }

    private void constructImpulseFunction(LayorInfo layorInfo)
    {
        //calculate y_k's aka output
        int len=0;
        for(Neuron neuron:layorInfo.layer)
        {
            layorInfo.vImpulseFunction.set(len, neuron.phi().apply(layorInfo.vInducedLocalField.get(len++)));
        }
    }

    protected void constructGradients(int example)
    {
       constructGradients(example, aLayers.length - 1);
    }

    protected void constructGradients(int example, int layer)
    {
        if (layer >= 0)
        {
            LayorInfo layorInfo = aLayers[layer];

            for(int neuronPos=0; neuronPos<layorInfo.layer.getNumberOfNeurons(); neuronPos++)
            {
                layorInfo.vGradients.set(neuronPos, gradient(example, layer, neuronPos));
            }

            constructGradients(example, layer - 1);
        }
    }

    /**
     * Gradient for the given example, layer, and neuron
     *
     * @param example example
     * @param layerLevel layer level
     * @param neuron neuron
     * @return gradient value for the given example, layer, and neuron
     */
    protected double gradient(int example, int layerLevel, int neuron)
    {
        if (layerLevel == aLayers.length-1)
        {
            // (oj - tj) * phi'_j(v^L_j)
            final double inducedLocalField = aLayers[layerLevel].vInducedLocalField.get(neuron);
            final double impulseFunction = aLayers[layerLevel].vImpulseFunction.get(neuron);
            return (aExamples[example].vExpected.get(neuron) - impulseFunction)
                    * phi.derivative(inducedLocalField);
        }
        else
        {
             return phi.derivative( aLayers[layerLevel].vInducedLocalField.get(neuron))
                     * sumGradients(example, layerLevel + 1, neuron);
        }
    }

    /**
     * Find the sum of the gradients times the weights of the next layer
     * for the current neuron
     *
     *
     *
     * @param example
     * @param layerLevel layer level
     * @param currentNeuron current neuron
     * @return
     */
    protected double sumGradients(int example, int layerLevel, int currentNeuron)
    {
        LayorInfo layorInfo = aLayers[layerLevel];
        double rslt = 0f;

        int neuronPos = 0;
        for(Neuron neuron:layorInfo.layer)
        {
            rslt += neuron.getWeight(currentNeuron) * gradient(example, layerLevel, neuronPos++);
        }
        return rslt;
    }

    protected void saveWeightAdjustments()
    {
        saveWeightAdjustments(aLayers.length - 1);
    }

    protected void saveWeightAdjustments(int layer)
    {
        if (layer >= 0)
        {
            LayorInfo layorInfo = aLayers[layer];

            int neuronPos = 0;
            for(Neuron neuron:layorInfo.layer)
            {
                //iterate thru the neuron's weights
                for(int weight=0; weight<neuron.getNumberOfWeights(); weight++)
                {
                    double weightAdjustment =
                            eta * layorInfo.vGradients.get(neuronPos) * layorInfo.vInput.get(weight); //delta correction

                    //save weight adjustment
                    double curWeightAdjustment = layorInfo.aWeightAdjustments[neuronPos].get(weight);
                    layorInfo.aWeightAdjustments[neuronPos].set(weight, curWeightAdjustment + weightAdjustment);
                }

                neuronPos++;
            }

            saveWeightAdjustments(layer - 1);
        }
    }

    protected void adjustWeights()
    {
        for(LayorInfo layorInfo:aLayers)
        {
            //iterate thru the neurons in the layer
            int neuronPos = 0;
            for(Neuron neuron:layorInfo.layer)
            {
                //iterate thru the neuron's weights
                for(int weight=0; weight<neuron.getNumberOfWeights(); weight++)
                {
                    //adjust weights
                    double momentum = alpha * layorInfo.aPrevWeights[neuronPos].get(weight);
                    double curWeight = neuron.getWeight(weight);

                    //backup previous weight
                    layorInfo.aPrevWeights[neuronPos].set(weight, neuron.getWeight(weight));

                    neuron.setWeight(weight,
                                     curWeight
                                     + momentum
                                     + layorInfo.aWeightAdjustments[neuronPos].get(weight) // total delta weights
                    );
                }
                neuronPos++;
            }
        }
    }

    protected void resetWeightAdjustments()
    {
        for(LayorInfo layorInfo:aLayers)
        {
            //iterate thru the neurons in the layer
            int neuronPos = 0;
            for(Neuron neuron:layorInfo.layer)
            {
                //iterate thru the neuron's weights
                for(int weight=0; weight<neuron.getNumberOfWeights(); weight++)
                {
                    layorInfo.aWeightAdjustments[neuronPos].set(weight, 0);
                }
                neuronPos++;
            }
        }
    }

    public NVector rawoutput(NVector input)
    {
        throw new NotImplementedException();
    }

//    @Override
//    public int getNumberOfNeurons()
//    {
//        return 3;
//    }


}
