---
layout:     post
title:      "dubbo系列 - telnet"
subtitle:   "不适合人类阅读，非常水的自我笔记"
date:       2019-07-13
author:     "Gary"
header-img: "img/post-bg-unix-linux.jpg"
tags:
---

# dubbo系列 - telnet

`Dubbo`服务器判断如果接受到的请求时候`String`类型时，会调到`TelnetHandlerAdapter#telnet`。

```java
public class TelnetHandlerAdapter extends ChannelHandlerAdapter implements TelnetHandler {

    private final ExtensionLoader<TelnetHandler> extensionLoader = ExtensionLoader.getExtensionLoader(TelnetHandler.class);

    @Override
    public String telnet(Channel channel, String message) throws RemotingException {
        //获取prompt，默认为dubbo>
        String prompt = channel.getUrl().getParameterAndDecoded(Constants.PROMPT_KEY, Constants.DEFAULT_PROMPT);
        boolean noprompt = message.contains("--no-prompt");
        message = message.replace("--no-prompt", "");
        StringBuilder buf = new StringBuilder();
        message = message.trim();
        String command;
        //如果有空格的话，第一个空格前的为command，后面为message
        if (message.length() > 0) {
            int i = message.indexOf(' ');
            if (i > 0) {
                command = message.substring(0, i).trim();
                message = message.substring(i + 1).trim();
            } else {
                command = message;
                message = "";
            }
        } else {
            command = "";
        }
        if (command.length() > 0) {
            if (extensionLoader.hasExtension(command)) {
                if (commandEnabled(channel.getUrl(), command)) {
                    try {
                        //通过SPI调用对应command的telnetHandler.telnet
                        String result = extensionLoader.getExtension(command).telnet(channel, message);
                        if (result == null) {
                            return null;
                        }
                        buf.append(result);
                    } catch (Throwable t) {
                        buf.append(t.getMessage());
                    }
                } else {
                    buf.append("Command: ");
                    buf.append(command);
                    buf.append(" disabled");
                }
            } else {
                buf.append("Unsupported command: ");
                buf.append(command);
            }
        }
        if (buf.length() > 0) {
            buf.append("\r\n");
        }
        //如果入参有带--no-prompt的话，不打印prompt
        if (prompt != null && prompt.length() > 0 && !noprompt) {
            buf.append(prompt);
        }
        return buf.toString();
    }

    private boolean commandEnabled(URL url, String command) {
        boolean commandEnable = false;
        String supportCommands = url.getParameter(Constants.TELNET);
        if (StringUtils.isEmpty(supportCommands)) {
            commandEnable = true;
        } else {
            String[] commands = Constants.COMMA_SPLIT_PATTERN.split(supportCommands);
            for (String c : commands) {
                if (command.equals(c)) {
                    commandEnable = true;
                    break;
                }
            }
        }
        return commandEnable;
    }

}
```

`TelnetHandlerAdapter`解析命令，然后通过SPI来调到具体的`TelnetHandler`实现类。

通过SPI的配置文件可以知道`Dubbo`一共有12个`TelnetHandler`扩展：


> clear=com.alibaba.dubbo.remoting.telnet.support.command.ClearTelnetHandler
> exit=com.alibaba.dubbo.remoting.telnet.support.command.ExitTelnetHandler
> help=com.alibaba.dubbo.remoting.telnet.support.command.HelpTelnetHandler
> status=com.alibaba.dubbo.remoting.telnet.support.command.StatusTelnetHandler
> log=com.alibaba.dubbo.remoting.telnet.support.command.LogTelnetHandler
> ls=com.alibaba.dubbo.rpc.protocol.dubbo.telnet.ListTelnetHandler
> ps=com.alibaba.dubbo.rpc.protocol.dubbo.telnet.PortTelnetHandler
> cd=com.alibaba.dubbo.rpc.protocol.dubbo.telnet.ChangeTelnetHandler
> pwd=com.alibaba.dubbo.rpc.protocol.dubbo.telnet.CurrentTelnetHandler
> invoke=com.alibaba.dubbo.rpc.protocol.dubbo.telnet.InvokeTelnetHandler
> trace=com.alibaba.dubbo.rpc.protocol.dubbo.telnet.TraceTelnetHandler
> count=com.alibaba.dubbo.rpc.protocol.dubbo.telnet.CountTelnetHandler

看一下`ListTelnetHandler`的实现：

```java
@Activate
@Help(parameter = "[-l] [service]", summary = "List services and methods.", detail = "List services and methods.")
public class ListTelnetHandler implements TelnetHandler {

    @Override
    public String telnet(Channel channel, String message) {
        StringBuilder buf = new StringBuilder();
        String service = null;
        boolean detail = false;
        //判断是否是指定具体的service
        if (message.length() > 0) {
            String[] parts = message.split("\\s+");
            for (String part : parts) {
                if ("-l".equals(part)) {
                    detail = true;
                } else {
                    if (service != null && service.length() > 0) {
                        return "Invaild parameter " + part;
                    }
                    service = part;
                }
            }
        } else {
            //判断是否之前有执行过cd的命令
            service = (String) channel.getAttribute(ChangeTelnetHandler.SERVICE_KEY);
            if (service != null && service.length() > 0) {
                buf.append("Use default service " + service + ".\r\n");
            }
        }
        //如果没有指定service，则获取DubboProtocol的Exporters
        if (service == null || service.length() == 0) {
            for (Exporter<?> exporter : DubboProtocol.getDubboProtocol().getExporters()) {
                if (buf.length() > 0) {
                    buf.append("\r\n");
                }
                buf.append(exporter.getInvoker().getInterface().getName());
                if (detail) {
                    buf.append(" -> ");
                    buf.append(exporter.getInvoker().getUrl());
                }
            }
        } else {
        //如果指定具体的service，则通过exporter找到具体的class对象，然后通过反射的方式获取method。
            Invoker<?> invoker = null;
            for (Exporter<?> exporter : DubboProtocol.getDubboProtocol().getExporters()) {
                if (service.equals(exporter.getInvoker().getInterface().getSimpleName())
                        || service.equals(exporter.getInvoker().getInterface().getName())
                        || service.equals(exporter.getInvoker().getUrl().getPath())) {
                    invoker = exporter.getInvoker();
                    break;
                }
            }
            if (invoker != null) {
                Method[] methods = invoker.getInterface().getMethods();
                for (Method method : methods) {
                    if (buf.length() > 0) {
                        buf.append("\r\n");
                    }
                    if (detail) {
                        buf.append(ReflectUtils.getName(method));
                    } else {
                        buf.append(method.getName());
                    }
                }
            } else {
                buf.append("No such service " + service);
            }
        }
        return buf.toString();
    }

}

```
