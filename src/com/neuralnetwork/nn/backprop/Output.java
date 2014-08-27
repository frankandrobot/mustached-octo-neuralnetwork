package com.neuralnetwork.nn.backprop;

import java.util.logging.Level;
import java.util.logging.Logger;

class Output
{
    private final static Logger LOGGER = Logger.getLogger(NNBackprop.class.getName());

    Output()
    {
        LOGGER.setLevel(Level.INFO);
    }

    public void ouput(int iteration, double error)
    {
        LOGGER.info("==================================================================");
        LOGGER.info("iteration: "+iteration);

        String num = String.format("%.10f", error);
        LOGGER.info("The new error is "+num);
    }
}
