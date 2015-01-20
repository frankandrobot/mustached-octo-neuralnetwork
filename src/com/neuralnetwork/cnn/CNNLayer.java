package com.neuralnetwork.cnn;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;

/**
 * Not to be confused with {@link com.neuralnetwork.nn.layer.NNLayer}
 *
 * A single CNNLayer is composed of several {@link com.neuralnetwork.core.interfaces.ICnnMap}
 *
 */
class CNNLayer {

    List<MapNode> lMaps = new LinkedList<MapNode>();


    public HashSet<MapNode> getInputMaps()
    {
        HashSet<MapNode> hsInputMaps = new HashSet<MapNode>();

        for(MapNode mapNode :lMaps)
            hsInputMaps.addAll(mapNode.getParents());

        return hsInputMaps;
    }
}
