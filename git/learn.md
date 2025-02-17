|建立新的分支(`feature-branch`)|`git checkout -b feature-branch`|
|確認目前在哪個分支(有加*號)|git branch|
|刪除本地分支(`feature-branch`)|先切回main:`git checkout main`；在執行:`git branch -d feature-branch`|
|刪除遠端分支(若已推送到github)| `git push origin --delete feature-branch`|

