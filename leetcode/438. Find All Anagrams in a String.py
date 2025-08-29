from collections import Counter, defaultdict
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        cnt = Counter(p)
        window = Counter()
        n = len(s)
        m = len(p)
        res = [] 
        matches = 0  # how many chars currently match cnt exactly
        for i in range(n):
            window[s[i]]+=1
            if window[s[i]] == cnt[s[i]]:
                matches+=1
            elif window[s[i]] == cnt[s[i]]+1:
                matches-=1
            if i >=m:
                left = s[i-m]
                if window[left] == cnt[left]:
                    matches -= 1
                elif window[left] == cnt[left]+1:
                    matches += 1
                window[left] -= 1
            if matches == len(cnt):
                res.append(i-m+1)
        return res 