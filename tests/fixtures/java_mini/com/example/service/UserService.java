package com.example.service;

import com.example.model.User;
import com.example.util.Strings;

public class UserService {

    private static final int MAX_NAME = 32;

    public boolean validate(User user) {
        if (user == null) {
            return false;
        }
        if (user.getName() == null || user.getName().isEmpty()) {
            return false;
        }
        user.setName(Strings.truncate(user.getName(), MAX_NAME));
        return true;
    }

    public String summarize(User user, String extra, int flags) {
        StringBuilder sb = new StringBuilder();
        sb.append("user=").append(user == null ? "null" : user.getName());
        sb.append(",extra=").append(extra == null ? "-" : extra);
        sb.append(",flags=").append(flags);
        sb.append(",thread=").append(Thread.currentThread().getName());
        sb.append(",vm=").append(System.getProperty("java.vm.name"));
        for (int i = 0; i < 24; i++) {
            sb.append(";step").append(i);
            if (i % 4 == 0) {
                sb.append("collect");
            } else if (i % 4 == 1) {
                sb.append("merge");
            } else if (i % 4 == 2) {
                sb.append("dedupe");
            } else {
                sb.append("report");
            }
        }
        String joined = sb.toString();
        String safeUser = (user == null || user.getName() == null) ? "anonymous" : user.getName();
        String safeExtra = (extra == null || extra.isEmpty()) ? "none" : extra.trim().toLowerCase();
        return "summary[" + safeUser + "," + safeExtra + "," + flags + "] " + joined;
    }
}
