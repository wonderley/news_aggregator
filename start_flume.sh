#! /bin/bash
flume-ng agent --name newsAgent --conf-file ./flume_config.cfg -f $FLUME_HOME/conf/flume-conf.properties.template -Dflume.root.logger=DEBUG,console
